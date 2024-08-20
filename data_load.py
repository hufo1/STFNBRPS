from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class EEGdataset(Dataset):

    def __init__(self, data_path, label_path):
        self.data = np.load(data_path).astype(np.float32)
        self.label = np.load(label_path).astype("int64")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    path = r"D:\Desktop\eeg_model\data/"


    traindatasets = EEGdataset(path+'data_train.npy', path+'label_train.npy')  # 初始化
    testdatasets = EEGdataset(path+'label_test.npy', path+'label_test.npy')  # 初始化

    train_loader = DataLoader(dataset=traindatasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=testdatasets, batch_size=64, shuffle=True)
    #
    print(len(traindatasets))
    print(len(testdatasets))
    for data,label in train_loader:
        print(data.shape)
        print(label.dtype)
        break