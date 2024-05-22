import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# 数据（分为初始化init、取数据getitem、长度len三部分）
class Covid_dataset(Dataset):

    def __init__(self, file_path, mode, col_idx):
        with open(file_path, "r") as f:
            data = list(csv.reader(f))
            data = np.array(data)
            data = data[1:, 1:]

            # 从训练集中取一部分做验证集，这里采用每五取一的方式
            if mode == "train":
                indices = [i for i in range(len(data)) if i % 5 != 0]
                x = data[indices, :].astype(float)
                x = x[:, col_idx]
                y = data[indices, -1].astype(float)
                self.y = torch.tensor(y)

            if mode == "val":
                indices = [i for i in range(len(data)) if i % 5 == 0]
                x = data[indices, :].astype(float)
                x = x[:, col_idx]
                y = data[indices, -1].astype(float)
                self.y = torch.tensor(y)

            if mode == "test":
                x = data[:, col_idx].astype(float)

            x = torch.tensor(x)
            self.x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)  # 对数据进行归一化
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
        self.fc1 = nn.Linear(dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # 添加残差块
        residual = x
        x = self.fc3(x)
        x = x + residual
        x = self.relu(x)
        x = self.fc4(x)

        # 把x张量的维度降低1
        if len(x.size()) > 1:
            x = x.squeeze(dim=1)
        return x

# 训练和验证函数
def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []
    min_val_loss = 999999999

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        # 开始训练
        model.train()
        for batch_x, batch_y in train_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_batch_loss = loss(pred, y)
            train_batch_loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数w,b
            optimizer.zero_grad()  # 梯度清0
            train_loss += train_batch_loss.cpu().item()  # train_batch_loss表示一个batch的loss的和。train_loss是这轮训练中，所有batch的loss的和
        plt_train_loss.append(train_loss / train_loader.dataset.__len__())  # 计算的是这一轮训练的平均loss

        # 开始验证
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, y = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, y)
                val_loss += val_batch_loss.cpu().item()  #  val_loss表示的是这个模型在这轮训练中，整个验证集上的loss的和
        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        if val_loss < min_val_loss:
            torch.save(model, save_path)
            min_val_loss = val_loss

        #  一轮结束后，输出这一轮的结果
        print("[%03d/%03d] %2.2f secs TrainLoss: %.6f ValLoss: %.6f"%(epoch+1, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend("train", "val")
    plt.show()

# 测试函数
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


def get_feature_importance(feature_data, label_data, k=4, column=False):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    这个函数的目的是， 找到所有的特征中， 比较有用的k个特征， 并打印这些列的名字。
    """
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。

    if column:                            # 如果需要打印选中的列
        k_best_features = [column[i+1] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ', k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。





device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    "lr": 0.0001,
    "momentum": 0.9,
    "epochs": 50,
    "save_path": "model_save/model.pth",
    "rel_path": "pred.csv"
}

train_file = "covid.train.csv"
test_file = "covid.test.csv"


# 读数据
data = pd.read_csv(train_file)
data = data.values.tolist()
data = np.array(data).astype(str)
data = data[:, 1:]

all_feature = True
if all_feature:
    col_idx = [i for i in range(0, 93)]
    dim = 93
else:
    _, col_idx = get_feature_importance(data[:, 0:-1], data[:, -1], k=16)
    dim = 16

train_data = Covid_dataset(train_file, "train", col_idx)
val_data = Covid_dataset(train_file, "val", col_idx)
test_data = Covid_dataset(test_file, "test", col_idx)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# 模型、损失函数、优化器实例化
model = myModel(dim=dim)
loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), config["lr"], weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])
evaluate(config["save_path"], device, test_loader, config["rel_path"])
