import pywt
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from st_gcn import *

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

class Conv2d_res(nn.Module):

    def __init__(self, in_channels, out_channels, residual=True, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        return mish(x + res)



class scnn_gcn(torch.nn.Module):
    def __init__(self, in_channels, class_num):
        super(scnn_gcn, self).__init__()
        self.eeg = Model(1, 128)

        self.residual = nn.Sequential(  # 结构残差
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        # 解码
        self.decode = nn.Sequential(
            Conv2d_res(256, 256),  # T, V -> T//2, V//2
            Conv2d_res(256, 128),  # T, V -> T//2, V//2
            Conv2d_res(128, 64),  # T, V -> T//2, V//2
            # Conv2d_res(128, class_num),  # T, V -> T//2, V//2
        )

        self.SoftMax = nn.Softmax(dim=1)


    def forward(self, x):
        #eeg-gcn
        # x = x.unsqueeze(1)
        x11 = x
        # print(x1.shape)
        x1 = self.eeg(x11) 
        
        return x1  


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')

    x = torch.rand(64,1,128*5,14).to(device)
    model = scnn_gcn(1, 2).to(device)

    y = model(x)

    print(y.shape)