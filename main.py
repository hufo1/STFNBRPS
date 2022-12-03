from model import MyFPN


import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Function for moving tensor or model to GPU
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cpu()
        else:
            return [x.cpu() for x in xs]
    else:
        return xs


if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # mode = MyFPN().to(device)
    mode = MyXT().to(device)
    # mode = RE_FPN().to(device)
    # mode = MyXTvit().to(device)
    # loss = tf.keras.losses.categorical_crossentropy
    criterion = nn.CrossEntropyLoss().to(device)
    # T_epochs = 40
    # T_list = []
    #
    # LR_SCHEDULE = [(3, 0.000075),
    #                (8, 0.00005),
    #                (15, 0.000025),
    #                (25, 0.0000125),
    #                (35, 0.00000625),
    #                ]
    #
    #
    # def lr_schedule(epoch, lr):
    #     if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
    #         return lr
    #     for i in range(len(LR_SCHEDULE)):
    #         if epoch == LR_SCHEDULE[i][0]:
    #             return LR_SCHEDULE[i][1]
    #     return lr


    # %%
    optimizer = torch.optim.Adam(mode.parameters(), lr=0.001)
    # optimizer_res_net = torch.optim.Adam([
    #     {"params": mode.fc[0].parameters(), "lr": 0.001},
    #     {"params": mode.fc[2].parameters(), "lr": 0.001},
    #     {"params": mode.fc[4].parameters(), "lr": 0.001},
    # ],
    #     lr=0.0001, betas=(0.9, 0.999))
    # 读入数据
    EMGTRAIN = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\processed_data\all_data\EMG_train_min0_u_law.npy'))
    # EMGVAL = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\s1\EMG_test.npy'))
    EMGTEST = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\processed_data\all_data\EMG_test_min0_u_law.npy'))

    LABELTRAIN = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\processed_data\all_data\label_train_min0_u_law.npy'))
    # LABELVAL = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\s1\label_test.npy'))
    LABELTEST = torch.from_numpy(np.load(r'C:\YFF\divide_NinaPro_database_5-master\processed_data\all_data\label_test_min0_u_law.npy'))
    # a=torch.from_numpy(EMGTRAIN)
    # # 变成4维
    EMGTRAIN = torch.unsqueeze(EMGTRAIN, 1)
    # EMGVAL = torch.unsqueeze(EMGVAL, 1)
    EMGTEST = torch.unsqueeze(EMGTEST, 1)



    traindatasets = Mydataset(EMGTRAIN.float(), LABELTRAIN.long())  # 初始化
    testdatasets = Mydataset(EMGTEST.float(), LABELTEST.long())
    train_loader = data.DataLoader(traindatasets, batch_size=66)
    test_loader = data.DataLoader(testdatasets, batch_size=66)


    train(mode, train_loader, criterion, optimizer, test_loader, 60)