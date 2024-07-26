import os
from typing import Optional,Tuple,List,Union,Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

# 设置GPU还是CPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""加载数据集"""
data = np.load('tiny_nerf_data.npz') # 加载数据集
images = data['images']  # 图像数据
poses = data['poses']  # 位姿数据
focal = data['focal']  # 焦距数值

height, width = images.shape[1:3]
near, far = 2., 6.
n_training = 100 # 训练数据数量
testimg_idx = 101 # 测试数据下标
testimg, testpose = images[testimg_idx], poses[testimg_idx]
plt.imshow(testimg) #显示测试图像‘testing’

"""从相机位姿数据中提取方向和原点数据"""
# 方向数据
dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
#首先遍历'poses'中的每个相机位姿'pose'，然后'pose[:3,:3]'提取位姿矩阵的前3*3部分，即相机的旋转矩阵。
#然后将旋转矩阵与向量[0,0,-1]相乘，将相机在本地坐标系中的视线方向(负Z轴)转换到世界坐标系中，得到相机在世界坐标系中的方向。
#对上面的方向在最后一个维度求和，得到每个相机的方向向量，最后将所有相机方向向量堆叠成一个数组，形状为'(num_poses, 3)'，其中前者为相机数量
# 原点数据
origins = poses[:, :3, -1] #提取每个相机位姿矩阵的左后一列的前3行，即相机的位置（原点）

"""使用3D箭头绘制每个相机在3D空间中的位置和方向，可视化相机的分布和方向"""
ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
_ = ax.quiver(origins[..., 0].flatten(),origins[..., 1].flatten(),origins[..., 2].flatten(),dirs[..., 0].flatten(),dirs[..., 1].flatten(), dirs[..., 2].flatten(), length=0.5, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('z')
plt.show()