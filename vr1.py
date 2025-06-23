import torch
import random

# 生成一个形状为 [5, 1280, 1280] 的随机张量
tensor = torch.rand(5, 1280, 1280)

# 定义窗口大小
window_size = 64

# 计算每个维度上的窗口数量
num_windows = tensor.shape[1] // window_size

# 计算总的窗口数量
total_windows = num_windows * num_windows

# 计算需要遮挡的窗口数量（80%）
num_windows_to_mask = int(total_windows * 0.8)

# 随机选择需要遮挡的窗口索引
windows_to_mask = random.sample(range(total_windows), num_windows_to_mask)

# 遍历张量的每一个窗口并进行遮挡处理
for window_index in windows_to_mask:
    row = (window_index // num_windows) * window_size
    col = (window_index % num_windows) * window_size
    # 遮挡窗口，将其值设为0
    tensor[:, row:row + window_size, col:col + window_size] = 0
tensor=tensor.view(tensor.shape[0],-1,tensor.shape[1],tensor.shape[2])
print(tensor.shape)
print("Masking completed.")
