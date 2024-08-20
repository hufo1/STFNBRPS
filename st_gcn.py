import torch
import torch.nn.functional as F
import torch.nn as nn
from graph import Graph

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, graph_args={}, node_num=14,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        graph_size = self.graph.get_size(node_num)

        # build networks
        spatial_kernel_size = graph_size[0]
        temporal_kernel_size = 5     # 时间纬度感受野
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * node_num)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((

            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs),
            
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(graph_size))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(1, out_channels, kernel_size=1)
        # self.SoftMax = nn.Softmax(dim=1)

    def forward(self, x):
        N, C, T, V = x.size()
        device = x.device
        A = self.graph(x).to(device)
        x = x.transpose(2,3).contiguous()    # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).transpose(2,3).contiguous()    # N, C, T, V

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, A * importance)

        x = self.fcn(x)


        return x

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,       # 时间维度上的感知野,邻接矩阵的分区数
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:#残差
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        # Normalize the adjacency matrix A to L
        L = normalize_adjacency(A)

        # Graph Convolution with normalized adjacency matrix
        s = self.gcn(x, L)
        
        # Apply ReLU activation
        return s, A
    
def normalize_adjacency(A):
    Dl = torch.sum(A, dim=-1, keepdim=True)  # Sum of each row (degree matrix)
    D_inv_sqrt = torch.pow(Dl, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    normalized_A = D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)
    return normalized_A

class ConvTemporalGraphical(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, L):
       
        x = torch.matmul(L, x.permute(0, 1, 3, 2))

        # Apply ReLU activation
        x = F.relu(x.permute(0, 1, 3, 2))

        return x

if __name__ == '__main__':
    x1 = torch.rand(64, 1, 128*5, 14)
    mo = Model(1, 3)
    y = mo(x1)
    print(y.shape)