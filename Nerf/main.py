import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from utils.model import NeRF
from utils.utils_algo import encode_position, get_rays,nerf_forward

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

"""优化器"""
def get_optimizer(models, lr=5e-4):
    trainable_parameters = []
    for model in models:
        trainable_parameters += list(model.parameters())
    return torch.optim.Adam(params=trainable_parameters, lr=lr, betas=(0.9, 0.999))

"""学习率调度器"""
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

"""损失函数"""
def get_loss_fn():
    loss_fn = nn.MSELoss(reduction='mean')
    return loss_fn

"""绘制图像"""
def show_image(rgb_map):
    plt.imshow(rgb_map.detach().cpu().numpy())
    plt.show()

"""训练循环"""
def train_nerf():
    H, W, focal = 400, 400, 555.0
    near, far = 2.0, 6.0
    n_samples, n_samples_hierarchical = 64, 64
    n_iters = 1000

    encoding_fn = lambda x: encode_position(x, num_frequencies=10)
    viewdirs_encoding_fn = lambda x: encode_position(x, num_frequencies=4)

    d_input = 3 + 3 * 2 * 10
    d_input_viewdirs = 3 + 3 * 2 * 4

    coarse_model = NeRF(d_input=d_input, d_viewdirs=d_input_viewdirs)
    fine_model = NeRF(d_input=d_input, d_viewdirs=d_input_viewdirs)

    optimizer = get_optimizer([coarse_model, fine_model])
    scheduler = get_scheduler(optimizer)
    loss_fn = get_loss_fn()

    for i in trange(n_iters):
        pose = torch.eye(4)

        rays_o, rays_d = get_rays(H, W, focal, pose)
        rgb_coarse, depth_coarse, acc_coarse, rgb_fine, depth_fine, acc_fine = nerf_forward(
            rays_o, rays_d, near, far, encoding_fn, coarse_model,
            n_samples=n_samples, fine_model=fine_model, n_samples_hierarchical=n_samples_hierarchical,
            viewdirs_encoding_fn=viewdirs_encoding_fn)

        target_rgb = torch.rand((H, W, 3))

        loss = loss_fn(rgb_coarse, target_rgb)
        if rgb_fine is not None:
            loss += loss_fn(rgb_fine, target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            show_image(rgb_coarse)
            if rgb_fine is not None:
                show_image(rgb_fine)

if __name__ == "__main__":
    train_nerf()