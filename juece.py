import torch
import torch.nn as nn
import torch.nn.functional as F
from mstff_for import Double_mstff
from scnn_for import scnn_gcn
from crosstransformer2 import Cross2
from ffm import FFM_model


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

class trans_fusion(nn.Module):
    def __init__(self, in_feature, class_num, graph_args={}, frames=128*5, node_num=14, edge_importance_weighting=True, dropout=0):
        super(trans_fusion, self).__init__()
        self.dmst = Double_mstff(in_feature, class_num, frames=frames, node_num=node_num)#time
        self.dsc_gcn = scnn_gcn(in_feature, class_num)#gcn
        self.ffm = FFM_model(256,256, node_num=node_num)
        # self.cross_trans2 = Cross2(19, 256)

        self.residual = nn.Sequential(  # 结构残差
            nn.Conv2d(in_feature, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        self.decode =  nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.flattened_size = 64 * frames * node_num
        self.fc1 = nn.Linear(self.flattened_size, 3)
       

        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, xinput):
        data = xinput.unsqueeze(1)  # 数据预处理[N, T, V]->[N, C, T, V]     
        x1 = self.dmst(data) 
        x1 = x1.permute(0, 1, 3, 2)
        x2 = self.dsc_gcn(data) 
        x2 = x2.permute(0, 1, 3, 2)

        y = torch.cat([x1, x2], dim = 1)
        y = y.permute(0, 3, 1, 2)
        N1, T1, C1, V1 = y.size()
        y1 = y.reshape(N1*T1, C1, V1)

        x3 = self.ffm(y1)
        N, T, C, V = x3.size()
        x3 = x3.permute(0, 2, 1, 3)
        #class
        z = self.decode(x3)
        z = z.view(z.size(0), -1)
        peo = self.fc1(z)
        # result1 = peo.argmax(1)
        # result = self.SoftMax(peo)

        return peo

        
if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # device = torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(64, 1, 128*5, 14).to(device)
    mo = trans_fusion(1, 3).to(device)
    print(mo(x1).shape)
