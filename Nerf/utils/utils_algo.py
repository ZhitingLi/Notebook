import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

"""定义位置编码器"""
def encode_position(x, num_frequencies=10):
    frequencies = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    encoding = [x]
    for freq in frequencies:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(x * freq))
    return torch.cat(encoding, dim=-1)

"""将一个大张量划分为多个较小的块"""
def get_chunks(
  inputs: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  Divide an input into chunks.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

"""准备射线的批次"""
def prepare_chunks(points, encoding_fn, chunksize):
    points = points.view(-1, points.shape[-1])
    encoded_points = encoding_fn(points)
    return torch.split(encoded_points, chunksize)

def prepare_viewdirs_chunks(points, rays_d, encoding_fn, chunksize):
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None].expand(points.shape).contiguous()
    encoded_viewdirs = encoding_fn(viewdirs.view(-1, viewdirs.shape[-1]))
    return torch.split(encoded_viewdirs, chunksize)

"""获取射线起点和方向"""
def get_rays(height, width, focal, pose):
    i, j = torch.meshgrid(torch.linspace(0, width - 1, width), torch.linspace(0, height - 1, height))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - width * 0.5) / focal, -(j - height * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

"""分层抽样：在已有的样本点基础上，进一步从概率密度中进行更多的采样，以提高采样的精度"""
def sample_hierarchical(rays_o, rays_d, z_vals, weights, n_samples, **kwargs):
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, **kwargs)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    query_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return query_points, z_vals, z_samples
"""分层体积采样"""
def sample_pdf(bins, weights, N_samples, det=False, eps=1e-5):
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1).view(list(inds.shape) + [2])
    cdf_g = torch.gather(cdf.unsqueeze(-1).expand(list(cdf.shape) + [2]), -2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-1).expand(list(bins.shape) + [2]), -2, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

"""将估计结果转化为图像"""
def raw2outputs(raw, z_vals, rays_d):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-raw[..., 3] * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    rgb = torch.sum(weights[..., None] * torch.sigmoid(raw[..., :3]), -2)
    depth = torch.sum(weights * z_vals, -1)
    acc = torch.sum(weights, -1)
    return rgb, depth, acc, weights

"""前向传播"""
def nerf_forward(rays_o, rays_d, near, far, encoding_fn, coarse_model, n_samples=64, fine_model=None, n_samples_hierarchical=64, viewdirs_encoding_fn=None, **kwargs_sample_hierarchical):
    t_vals = torch.linspace(0., 1., steps=n_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand
    query_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    batches = prepare_chunks(query_points, encoding_fn, chunksize=1024 * 32)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=1024 * 32)
    else:
        batches_viewdirs = [None] * len(batches)

    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0).reshape(list(query_points.shape[:-1]) + [predictions[0].shape[-1]])

    rgb_coarse, depth_coarse, acc_coarse, weights = raw2outputs(raw, z_vals, rays_d)

    if n_samples_hierarchical > 0:
        query_points, z_vals_combined, new_z_samples = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical)

        batches = prepare_chunks(query_points, encoding_fn, chunksize=1024 * 32)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn, chunksize=1024 * 32)
        else:
            batches_viewdirs = [None] * len(batches)

        fine_predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            fine_predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(fine_predictions, dim=0).reshape(list(query_points.shape[:-1]) + [fine_predictions[0].shape[-1]])
        rgb_fine, depth_fine, acc_fine, _ = raw2outputs(raw, z_vals_combined, rays_d)
    else:
        rgb_fine, depth_fine, acc_fine = None, None, None

    return rgb_coarse, depth_coarse, acc_coarse, rgb_fine, depth_fine, acc_fine

"""训练类和函数"""
# 绘制采样函数
def plot_samples(
  z_vals: torch.Tensor,
  z_hierarch: Optional[torch.Tensor] = None,
  ax: Optional[np.ndarray] = None):
  r"""
  绘制分层样本和（可选）分级样本。
  """
  y_vals = 1 + np.zeros_like(z_vals)

  if ax is None:
    ax = plt.subplot()
  ax.plot(z_vals, y_vals, 'b-o')
  if z_hierarch is not None:
    y_hierarch = np.zeros_like(z_hierarch)
    ax.plot(z_hierarch, y_hierarch, 'r-o')
  ax.set_ylim([-1, 2])
  ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
  ax.axes.yaxis.set_visible(False)
  ax.grid(True)
  return ax

def crop_center(
  img: torch.Tensor,
  frac: float = 0.5
) -> torch.Tensor:
  r"""
  从图像中裁剪中心方形。
  """
  h_offset = round(img.shape[0] * (frac / 2))
  w_offset = round(img.shape[1] * (frac / 2))
  return img[h_offset:-h_offset, w_offset:-w_offset]

