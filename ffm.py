import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize(data, range_min=0, range_max=1):
    """
    将数据归一化到指定范围。

    参数:
    data (array-like): 需要归一化的数据。
    range_min (float): 归一化后的最小值（默认为0）。
    range_max (float): 归一化后的最大值（默认为1）。

    返回:
    normalized_data (np.ndarray): 归一化后的数据。
    """
    data = np.asarray(data)
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 防止除零错误
    if data_max == data_min:
        return np.full_like(data, range_min)

    # 归一化
    normalized_data = (data - data_min) / (data_max - data_min)
    
    # 缩放到指定范围
    normalized_data = normalized_data * (range_max - range_min) + range_min
    
    return normalized_data

def sigmoid(x):
    # 如果x是GPU上的张量，先将其移动到CPU
    if x.is_cuda:
        x = x.cpu()
    # 分离张量，使其不再与计算图连接
    x = x.detach()
    # normalized_data = normalize(x)
    # y = 1 / (1 + np.exp(-x))
    y = np.exp(-np.logaddexp(0, -x))

    return y

class FFM_model(nn.Module):
    def __init__(self, in_channels, out_channels, node_num):
        super().__init__()
        # 空间特征提取
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_1 =  nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=3//2),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=5//2),
            nn.ReLU()
        )
        self.out = node_num
        # self.fc1 = nn.Linear( self.NT * self.V, out_channels)



    def forward(self, x):
        # N, C, T, V = x.size()     # C = 1
        #全局
        x_g1 = self.conv_1(x)    # 分支一：N,C_out,T,V
        x_g2 = self.conv_2(x)    # 分支二：N,C_out,T,V
        x_g3 = self.conv_3(x)    # 分支三：N,C_out,T,V
        y = torch.add(torch.add(x_g1, x_g2), x_g3)
        y = y.permute(1, 0, 2)
        C, NT, V = y.size()
        y = torch.sum(y, dim=1)
        y1 = y.unsqueeze(0)
        y1 = y1.expand(NT, -1, -1)


        # 局部
        x_l = self.conv_1(x)
        y2 = torch.add(y1, x_l)
        y2 = y2.reshape(y1.size(2), -1)

        self.fc1 = nn.Linear(NT * C, self.out).to(self.device)
        z1 = sigmoid(self.fc1(y2))
        z1_tensor = z1.to(x.device)
        z1_tensor = z1_tensor.transpose(0, 1)

        x = x.reshape(-1, 128*5, C, V)
        X = torch.einsum('btcn,nk->btck', x, z1_tensor)
        


        return X            # N,C,T,V -> N,4*C_out,T,V
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')

    x = torch.rand(64*128*5,256,14).to(device)
    model = FFM_model(256,256,14).to(device)

    y = model(x)

    print(y.shape)