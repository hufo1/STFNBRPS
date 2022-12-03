import torch
import torch.nn.functional as F
import torch.nn as nn
import math
# from model.attention.SKAttention import SKAttention
# from model.attention.SEAttention import SEAttention
from torch.nn import init
##注意力机制
# class SpectralNorm(tf.keras.constraints.Constraint):
#     def __init__(self, n_iter=5):
#         self.n_iter = n_iter
#     def call(self, input_weights):
#         w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
#         u = tf.random.normal((w.shape[0], 1))
#         for _ in range(self.n_iter):
#             v = tf.matmul(w, u, transpose_a=True)
#             v /= tf.norm(v)
#             u = tf.matmul(w, v)
#             u /= tf.norm(u)
#         spec_norm = tf.matmul(u, tf.matmul(w, v),    transpose_a=True)
#         return input_weights/spec_norm
#
# class SelfAttention(tf.keras.layers.Layer):
#     def __init__(self,head):
#         super(SelfAttention, self).__init__()
#         self.head=head
#     def build(self, input_shape,heads):
#         n, h, w, c = input_shape
#         self.n_feats = h * w
#         self.conv_theta = tf.keras.layers.Conv2D(c //self.head  ,1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Theta')
#         self.conv_phi = tf.keras.layers.Conv2D(c // self.head, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Phi')
#         self.conv_g = tf.keras.layers.Conv2D(c // self.head, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_g')
#         self.conv_attn_g = tf.keras.layers.Conv2D(c , 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_AttnG')
#         self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma')
#     def call(self, x):
#         n, h, w, c = x.shape
#         theta = self.conv_theta(x)
#         theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))
#         phi = self.conv_phi(x)
#         phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
#         phi = tf.reshape(phi, (-1, self.n_feats//4, phi.shape[-1]))
#         g = self.conv_g(x)
#         g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
#         g = tf.reshape(g, (-1, self.n_feats//4, g.shape[-1]))
#         attn = tf.matmul(theta, phi, transpose_b=True)
#         attn = tf.nn.softmax(attn)
#         attn_g = tf.matmul(attn, g)
#         attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
#         attn_g = self.conv_attn_g(attn_g)
#         output = x + self.sigma * attn_g
#         return output

class BiLSTM(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(n_hidden * 2, n_class)
    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]
        hidden_state = torch.randn(1*2, batch_size,  self.n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size,  self.n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model

#
# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#         print("Mish activation loaded...")
#     def forward(self,x):
#         x = x * (torch.tanh(F.softplus(x)))
#         return x


def mish(x):
    return x*(torch.tanh(F.softplus(x)))


class XceptionTime_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
# def XceptionTime_model(input, f):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation=2)
        # x = tf.keras.layers.Conv2D(filters=f, kernel_size=1)(input)
        # x1 = tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', strides=1)(x)
        # x2 = tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', strides=1,dilation_rate=2)(x)
        # for i in range(2):
        #     self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation_rate=2)
        #     # x2 = tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', strides=1,dilation_rate=2)(x2)
        # x3 = tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', strides=1,dilation_rate=2)(x)
        # for i in range(4):
        #     x3= tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same', strides=1,dilation_rate=2)(x3)

        self.conv3 = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, padding='same')
    # y = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(input)
    # y = tf.squeeze(y, axis=3)
    # y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=f, return_sequences=True))(y)
        self.relu = nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(512)
        # self.liner1 = nn.Linear(in_features=896 * 190, out_features=10)#前加平均池化
        self.bn = torch.nn.BatchNorm1d(40)
        self.lstm = nn.LSTM(input_size=16, hidden_size=out_channels, bidirectional=True)
        self.conv1d= nn.Conv1d(in_channels=out_channels*2, out_channels=16, kernel_size=1)
        self.conv4 = nn.Conv2d(1, out_channels, kernel_size=3, padding='same')
        # self.bilstm = BiLSTM(in_channels, out_channels)
    # y = tf.nn.relu(y)
    # y = tf.keras.layers.BatchNormalization()(y)
    # y = tf.keras.layers.LocallyConnected1D(10, kernel_size=1, activation='relu')(y)
    # y = tf.keras.layers.BatchNormalization()(y)
    # y = tf.expand_dims(y, axis=3)
    # y = tf.keras.layers.Conv2D(filters=f, kernel_size=3, padding='same')(y)
    # sum1 = tf.keras.layers.concatenate([x1, x2, x3, y], axis=3)
    # sum1=SK_Net(sum1,f,16,4)
    # return sum1
    def forward(self, x0):
        x = self.conv(x0)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        for i in range(2):
            x2 = self.conv2(x2)
        x3 = self.conv2(x)
        for i in range(4):
            x3 = self.conv2(x3)
        y = self.conv3(x0)  #(50,1,40,16)
        y = torch.squeeze(y, 1)
        y=y# y = self.bilstm(y)
        y, (h, c) = self.lstm(y)#(50,40,32)20,10=20,32
        # y = self.lstm(y.permute(0, 2, 1))
        y = self.relu(y)
        y = self.bn(y)
        y = self.conv1d(y.permute(0, 2, 1))  #(50,16,40)
        y = torch.unsqueeze(y.permute(0, 2, 1), 1)#(50,1,40,16)
        y = self.conv4(y)  #(50,16,40,16)
        sum = torch.cat([x1, x2, x3, y], 1)  #(50,64,40,16)
        # sum=sum.permute(0, 2, 3, 1)
        return sum


# def XceptionTime(input_):
#     # MyReLU()是创建一个MyReLU对象，
#     # Function类利用了Python __call__操作，使得可以直接使用对象调用__call__制定的方法
#     # __call__指定的方法是forward，因此下面这句MyReLU（）（input_）相当于
#     # return MyReLU().forward(input_)
#     return XceptionTime_model(1, 16)(input_)


class MyFPN(torch.nn.Module):
    def __init__(self):
        super(MyFPN, self).__init__()
        # self.mish = Mish()
        self.X = XceptionTime_model(1, 16)
        self.Z = nn.Conv2d(1, 64, kernel_size=1)
        self.Zy4 = nn.Conv2d(256, 64, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(64)
        # self.se = SEAttention(channel=64, reduction=16)

        self.X1 = XceptionTime_model(64, 32)
        self.Z1 = nn.Conv2d(64, out_channels=128, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        # self.se1 = SEAttention(channel=128, reduction=16)

        self.X2 = XceptionTime_model(128, 64)
        self.Z2 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm2d(256)
        # self.se2 = SEAttention(channel=256, reduction=16)

        self.X3 = XceptionTime_model(256, 128)
        self.Z3 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(512)
        # self.se3 = SEAttention(channel=512, reduction=16)

        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.p41 = nn.Conv2d(512, out_channels=256, kernel_size=1)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        self.p42 = nn.Conv2d(256, out_channels=256, kernel_size=3, padding='same')

        self.p31 = nn.Conv2d(256, out_channels=256, kernel_size=1)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        self.p32 = nn.Conv2d(256, out_channels=256, kernel_size=3, padding='same')

        self.p21 = nn.Conv2d(128, out_channels=256, kernel_size=1)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        self.p22 = nn.Conv2d(256, out_channels=256, kernel_size=3, padding='same')

        self.p11 = nn.Conv2d(64, out_channels=256, kernel_size=1)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        self.p12 = nn.Conv2d(256, out_channels=256, kernel_size=3, padding='same')

        self.conv = nn.Conv2d(1536, 512, kernel_size=1)

        self.conv256 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv128 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv52 = nn.Conv2d(128, 52, kernel_size=[40, 16])

        self.bn52 = torch.nn.BatchNorm2d(52)
        self.dropout = nn.Dropout(0.2)
    def forward(self, xinput):
        ##加上conv
        x = self.X(xinput)
        z = self.Z(xinput)
        z = self.bn(z)
        sum1 = mish(x+z)
        # sum1 = self.se(sum1)*sum1


        x1 = self.X1(sum1)
        z1 = self.Z1(sum1)
        z1 = self.bn1(z1)
        sum2 = mish(x1 + z1)
        # sum2 = self.se1(sum2) * sum2

        x2 = self.X2(sum2)
        z2 = self.Z2(sum2)
        z2 = self.bn2(z2)
        sum3 = mish(x2 + z2)
        # sum3 = self.se2(sum3) * sum3

        x3 = self.X3(sum3)
        z3 = self.Z3(sum3)
        z3 = self.bn3(z3)
        sum4 = mish(x3 + z3)
        # sum4 = self.se3(sum4) * sum4

        ##FPN
        p5_2 = self.max_pool(sum4)#???

        p4_1 = self.p41(sum4)
        p4_1 = self.bn2(p4_1)
        p4_2 = self.p42(p4_1)+p4_1

        p3_1 = self.p31(sum3)
        p3_1 = self.bn2(p4_1+p3_1)
        p3_2 = self.p32(p3_1) + p3_1

        p2_1 = self.p21(sum2)
        p2_1 = self.bn2(p3_1+p2_1)
        p2_2 = self.p22(p2_1) + p2_1

        p1_1 = self.p11(sum1)
        p1_1 = self.bn2(p2_1+p1_1)
        p1_2 = self.p12(p1_1) + p1_1   #(50,256.40,16)
        # # 无ASSP的RE_FPN
        #
        # x4 = self.X(xinput)
        # z4 = self.Z(xinput)
        # z4 = self.bn(z4)
        # y4 = self.Zy4(p1_2)
        # y4 = self.bn(y4)
        # sum5 = mish(self.sk(x4+y4)+z4)
        # sum1 = self.se(sum1) * sum1
        #
        # x1 = self.X1(sum1)
        # z1 = self.Z1(sum1)
        # z1 = self.bn1(z1)
        # sum2 = mish(x1 + z1)
        # sum2 = self.se1(sum2) * sum2
        #
        # x2 = self.X2(sum2)
        # z2 = self.Z2(sum2)
        # z2 = self.bn2(z2)
        # sum3 = mish(x2 + z2)
        # sum3 = self.se2(sum3) * sum3
        #
        # x3 = self.X3(sum3)
        # z3 = self.Z3(sum3)
        # z3 = self.bn3(z3)
        # sum4 = mish(x3 + z3)
        # sum4 = self.se3(sum4) * sum4

        # # p42=p41+p41
        # ##FPN
        # p5_2 = self.max_pool(sum4)  # ???
        #
        # p4_1 = self.p41(sum4)
        # p4_1 = self.bn2(p4_1)
        # p4_2 = self.p42(p4_1) + p4_1
        #
        # p3_1 = self.p31(sum3)
        # p3_1 = self.bn2(p4_1 + p3_1)
        # p3_2 = self.p32(p3_1) + p3_1
        #
        # p2_1 = self.p21(sum2)
        # p2_1 = self.bn2(p3_1 + p2_1)
        # p2_2 = self.p22(p2_1) + p2_1
        #
        # p1_1 = self.p11(sum1)
        # p1_1 = self.bn2(p2_1 + p1_1)
        # p1_2 = self.p12(p1_1) + p1_1  # (50,256.40,16)???

        sum9 = torch.cat([p1_2, p2_2, p3_2, p4_2, p5_2], 1)#1536
        sum9 = F.relu(self.conv(sum9))

        ##jiema
        z9 = self.conv256(sum9)
        z9 = self.bn2(z9)
        sum9 = self.conv256(sum9)
        sum9 = self.bn2(sum9)
        sum9 = mish(sum9 + z9)
        sum9 = self.dropout(sum9)

        # sum4=tf.keras.layers.LocallyConnected2D(filters=128, kernel_size=1)(sum4)
        z9 = self.conv128(sum9)
        z9 = self.bn1(z9)
        sum9 = self.conv128(sum9)
        sum9 = self.bn1(sum9)
        sum9 = mish(sum9 + z9)
        sum9 = self.dropout(sum9)

        sum9 = self.conv52(sum9)
        # sum4 = tf.keras.layers.Conv2D(filters=52, kernel_size=1)(sum4)
        sum9 = self.bn52(sum9)
        sum9 = mish(sum9)

        logits = torch.squeeze(sum9)##(50,52)
        logits = F.softmax(logits, dim=-1)

        return logits

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')

    x = torch.rand(50, 1, 40, 16).to(device)
    model = MyFPN().to(device)
    # model = MyXT().to(device)
    # model = CNN_2D().to(device)
    # model = XceptionTime_model().to(device)
    # # model = Cnn1d().to(device)
    y = model(x)
    print(y)
    # print(FPN101())