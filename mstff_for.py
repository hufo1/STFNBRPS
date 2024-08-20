import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

# XT
class XceptionTime_model(nn.Module):
    def __init__(self, in_channels, out_channels , frames, node_num):
        super().__init__()
        # 空间特征提取
        self.conv_s = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv_s1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_s2 = nn.Conv2d(16, 32, kernel_size=5, padding=5//2)
        self.conv_s3 = nn.Conv2d(16, 32, kernel_size=7, padding=7//2)
       
        # 时间特征提取
        hidden_size = 32
        self.conv_t1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.lstm = nn.LSTM(node_num, hidden_size, bidirectional=True, batch_first=True)
        self.bn_Re = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
        )
        self.conv_t2 = nn.Conv1d(hidden_size * 2, node_num, kernel_size=1)
        self.conv_t3 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.conv_t4 = nn.Conv2d(in_channels,128, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        N, C, T, V = x.size()     # C = 1
        # 空间
        x_s = self.conv_s(x)       
        x_s1 = self.conv_s1(x_s)    
        x_s2 = self.conv_s2(x_s)   
        x_s3 = self.conv_s3(x_s)   
        # 时间
        x_t = self.conv_t1(x)                                  
        x_t, _ = self.lstm(x_t.view(N, T, V))                   # N,C,T,V -> N,T,V -> N,T,hid_size
        x_t = self.bn_Re(x_t.transpose(1,2).contiguous())       # N,T,hid_size -> N,hid_size,T
        x_t = self.conv_t2(x_t)                                 # N,hid_size,T -> N,V,T
        x_t = x_t.transpose(1,2).contiguous().view(N, C, T, V)  # N,V,T -> N,C,T,V
        x_t = self.conv_t3(x_t)                                 # N,C,T,V -> N,C_out,T,V

        #concat
        y = torch.cat([x_s1, x_s2, x_s3, x_t], dim = 1)       
        #
        x_2d = self.conv_t4(x)
        
        _y = torch.add(y, x_2d)

        return _y          


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

#XT主函数
class Double_mstff(torch.nn.Module):
    def __init__(self, in_channels, out_channels, frames, node_num):
        super(Double_mstff, self).__init__()
        self.data_bn_eeg = nn.BatchNorm1d(in_channels * node_num)
        self.XT_eeg = XceptionTime_model(in_channels, 64, frames, node_num)

       
    def forward(self, x):
        # 数据预处理

        x1 = x
        N, C, T, V = x1.size()
        x1 = x1.transpose(2, 3).contiguous()  # N, C, V, T  
        x1 = x1.view(N, C * V, T)
        x1 = self.data_bn_eeg(x1)
        x1 = x1.view(N, C, V, T).transpose(2, 3).contiguous()  # N, C, T, V   
        x1 = self.XT_eeg(x1)
        # print(x1.shape)

       

        return x1      


if __name__ == '__main__':
    x1 = torch.rand(64, 1, 640, 14)
    mo = Double_mstff(1, 32, 32, 14)
    print(mo(x1).shape)