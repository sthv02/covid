import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
import time



class Covid_dataset(Dataset):
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:
            csv_data = list(csv.reader(f))
            data = np.array(csv_data)[1:]   # 把链表先变成数组，再去掉数据的第一行（不包括第一行）
                                            # [1:]（第一行后面的，不包括第一行）
                                            # [1:,1：-1]（第一行后面的，第一列和最后一列之间的，都是开区间）
                                            # [1：,-1](第一行之后的行和最后一列)

            if mode == "train":
                indices = [i for i in range(len(data)) if i % 5 != 0]
            elif mode == "val":
                indices = [i for i in range(len(data)) if i % 5 == 0]

            if mode == 'test':
                x = data[:, 1:].astype(float)
                x = torch.tensor(x)
            else:
                x = data[indices, 1:-1].astype(float)
                x = torch.tensor(x)
                y = data[indices, -1].astype(float)
                self.y = torch.tensor(y)

            self.x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0,keepdim=True)   # 对数据进行归一化

            self.mode = mode

    def __getitem__(self, item):
        if self.mode == "test":
            return self.x[item].float()
        else:
            return self.x[item].float(), self.y[item].float()

    def __len__(self):
        return len(self.x)


class myModel(nn.Module):
    def __init__(self, dim):
        super(myModel, self).__init__()
        self.fc1 = nn.Linear(dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            x = x.squeeze(dim=1)

        return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []
    min_val_loss = 99999999999

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)     # 正向传播的时候同时计算梯度
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()   # 梯度反向传播来进行优化
            optimizer.step()    # 更新参数
            optimizer.zero_grad()   # 梯度归0
            train_loss += train_bat_loss.cpu().item()

        plt_train_loss.append(train_loss / train_loader.dataset.__len__())

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)     # 正向传播的时候同时计算梯度
                val_bat_loss = loss(pred, target)
                val_loss += val_bat_loss.cpu().item()
        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        if val_loss < min_val_loss:
            torch.save(model, save_path)
            min_val_loss = val_loss

        print("[%03d/%03d] %2.2f secs Trainloss: %.6f Valloss: %.6f"%(epoch, epochs, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1]))


    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()


def evaluate(save_path, device, test_loader, rel_path):
    model = torch.load(save_path).to(device)
    rel = []
    model.eval()

    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())

    with open(rel_path, "w") as f:
        csv_writter = csv.writer(f)
        csv_writter.writerow(["id", "tested_positive"])
        for i in range(len(rel)):
            csv_writter.writerow([str(i), str(rel[i])])
        print("文件已经保存到"+rel_path)


# for batch_x, batch_y in train_loader:
#     print(batch_x, batch_y)
#     pred = model(batch_x)


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_data = Covid_dataset(train_file, "train")
val_data = Covid_dataset(train_file, "val")
test_data = Covid_dataset(test_file, "test")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # 一次取16个数据
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

dim = 93

config = {
    "lr": 0.001,
    "momentum": 0.9,
    "epochs": 20,
    "save_path": "model_save/model.pth",
    "rel_path": "pred.scv"
}

model = myModel(dim)

loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])

# evaluate(config["save_path"], device, test_loader, config["rel_path"])























