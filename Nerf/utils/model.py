import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

"""定义NeRF模型"""
class NeRF(nn.Module):
    def __init__(self, d_input, n_layers=8, d_filter=256, skip=4, d_viewdirs=None):
        super(NeRF, self).__init__()
        self.skip = skip
        self.d_viewdirs = d_viewdirs

        layers = [nn.Linear(d_input, d_filter)]
        for i in range(n_layers - 1):
            if i % skip == 0 and i > 0:
                layers.append(nn.Linear(d_filter + d_input, d_filter))
            else:
                layers.append(nn.Linear(d_filter, d_filter))
        self.pts_linears = nn.ModuleList(layers)

        if d_viewdirs is not None:
            self.bottleneck_linear = nn.Linear(d_filter, d_filter)
            self.view_linear = nn.Linear(d_filter + d_viewdirs, d_filter // 2)
            self.output_linear = nn.Linear(d_filter // 2, 4)
        else:
            self.output_linear = nn.Linear(d_filter, 4)

    def forward(self, x, viewdirs=None):
        """带有视图方向的向前传播"""
        h = x
        for i, l in enumerate(self.pts_linears):
            if i % self.skip == 0 and i > 0:
                h = torch.cat([x, h], -1)
            h = F.relu(l(h))

        if self.d_viewdirs is not None:
            bottleneck = self.bottleneck_linear(h)
            h = torch.cat([bottleneck, viewdirs], -1)
            h = F.relu(self.view_linear(h))

        outputs = self.output_linear(h)
        return outputs

"""体积渲染：将nerf的'raw'输出转化为图像的'RGB'颜色图、深度图、时差图和累积透明图"""
def cumprod_exclusive(
        tensor: torch.Tensor
) -> torch.Tensor:
    # 首先计算规则的cunprod
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    # 用1替换首个元素
    cumprod[..., 0] = 1.
    return cumprod

class EarlyStopping:
  r"""
  基于适配标准的早期停止辅助器
  """
  def __init__(
    self,
    patience: int = 30,
    margin: float = 1e-4
  ):
    self.best_fitness = 0.0
    self.best_iter = 0
    self.margin = margin
    self.patience = patience or float('inf')  # 在epoch停止提高后等待的停止时间

  def __call__(
    self,
    iter: int,
    fitness: float
  ):
    r"""
    检查是否符合停止标准。
    """
    if (fitness - self.best_fitness) > self.margin:
      self.best_iter = iter
      self.best_fitness = fitness
    delta = iter - self.best_iter
    stop = delta >= self.patience  # 超过耐性则停止训练
    return stop

def init_models():
  r"""
  为 NeRF 训练初始化模型、编码器和优化器。
  """
  # 编码器
  encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
  encode = lambda x: encoder(x)

  # 视图方向编码
  if use_viewdirs:
    encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                        log_space=log_space)
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output
  else:
    encode_viewdirs = None
    d_viewdirs = None

  # 模型
  model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
              d_viewdirs=d_viewdirs)
  model.to(device)
  model_params = list(model.parameters())
  if use_fine_model:
    fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                      d_viewdirs=d_viewdirs)
    fine_model.to(device)
    model_params = model_params + list(fine_model.parameters())
  else:
    fine_model = None

  # 优化器
  optimizer = torch.optim.Adam(model_params, lr=lr)
  # 早停
  warmup_stopper = EarlyStopping(patience=50)
  return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper
