# 1.导入必要的库
import torch
import pandas as pd
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time

# 超参数
BATCH_SIZE = 8
nepochs = 1000
LR = 0.001

# 定义损失函数为交叉熵损失 loss_func
loss_func = nn.CrossEntropyLoss()

# 可以在GPU或者CPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 2.CIFAR10数据集的预处理

# CIFAR10的输入图片各channel的均值 mean 和标准差 std
mean = [x / 255 for x in [125.3, 23.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
n_train_samples = 50000

# 如果是多进程需要加一个main函数，否则会报错
    # 数据增强-->训练集
train_set = dsets.CIFAR10(root='CIFAR10/',  # 数据集保存路径
                              train=True,
                              download=True,  # 如果未下载，改为True；如果已经下载好，改为False
                              transform=trans.Compose([
                                  trans.RandomHorizontalFlip(),
                                  trans.RandomCrop(32, padding=4),
                                  trans.ToTensor(),
                                  trans.Normalize(mean, std)
                              ]))
train_dl = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,)

# train_set.train_data = train_set.train_data[0:n_train_samples]
# train_set.train_labels = train_set.train_labels[0:n_train_samples]

    # 测试集
test_set = dsets.CIFAR10(root='CIFAR10/',  # 数据集保存路径
                             train=False,
                             download=True,  # 如果未下载，改为True；如果已经下载好，改为False
                             transform=trans.Compose([
                                 trans.ToTensor(),
                                 trans.Normalize(mean, std)
                             ]))

test_dl = DataLoader(test_set,
                         batch_size=BATCH_SIZE,)


# 3.定义训练的辅助函数，其中包括误差 error 与正确率 accuracy
def eval(model, loss_func, dataloader):
    model.eval()
    loss, accuracy = 0, 0
    id_array = []
    label_array = []
    k = 0
    # torch.no_grad显示地告诉pytorch，前向传播的时候不需要存储计算图
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y == batch_y.data).float().sum() / batch_y.size(0)
            pred_y = pred_y.tolist()
            #print("label:",pred_y)
            for i in range(BATCH_SIZE):
                k+=1
                id_array.append(k)
                label_array.append(pred_y[i])
            # print(label_array)

    loss /= len(dataloader)
    accuracy = accuracy * 100.0 / len(dataloader)
    return loss, accuracy, id_array, label_array



def train_epoch(model, loss_func, optimizer, dataloader):
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()

# 定义卷积层，在VGGNet中，均使用3x3的卷积核
def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

# 搭建VGG19，除了卷积层外，还包括2个全连接层（fc_1、fc_2），1个softmax层
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1.con1_1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2.con1_2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3.con2_1
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4.con2_2
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5.con3_1
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6.con3_2
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7.con3_3
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8.con3_4
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9.con4_1
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10.con4_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11.con4_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12.con4_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13.con5_1
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14.con5_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15.con5_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16.con5_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.classifier = nn.Sequential(
            # 17.fc_1
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18.fc_2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19.softmax
            nn.Linear(4096, 10),  # 最后通过softmax层，输出10个类别
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

vgg19 = VGG().to(device)
# 可以通过打印vgg19观察具体的网络结构
# print(vgg19)

# 使用Adam进行优化处理
optimizer = torch.optim.Adam(vgg19.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
learn_history = []
TRAIN = 0
if TRAIN:
    print('开始训练VGG19……')

    for epoch in range(nepochs):
        # 训练开始时间
        since = time.time()
        train_epoch(vgg19, loss_func, optimizer, train_dl)

        # 每训练5轮输出一次结果
        if (epoch) % 5 == 0:
            tr_loss, tr_acc, tr_id_array, tr_Forecast_label_array = eval(vgg19, loss_func, train_dl)
            te_loss, te_acc, te_id_array, te_Forecast_label_array = eval(vgg19, loss_func, test_dl)
            learn_history.append((tr_loss, tr_acc, te_loss, te_acc))
            # 完成一批次训练的结束时间
            now = time.time()
            print('[%3d/%d, %.0f seconds]|\t 训练误差: %.4f, 训练正确率: %.4f\t |\t 测试误差: %.4f, 测试正确率: %.4f' % (
                epoch, nepochs, now - since, tr_loss, tr_acc, te_loss, te_acc))
        torch.save(vgg19.state_dict(), "./model/vgg19_epoch{}.pt".format(epoch))

else:
    print("开始测试.....")
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    # data = unpickle("./CIFAR10/cifar-10-batches-py/test_batch")
    # print(data)
    label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    vgg19 = VGG().to(device)
    Forecast_label_list = []
    vgg19.load_state_dict(torch.load('./model/vgg19_epoch170.pt')) # 导入网络的参数
    test_loss, test_accuracy, id_array, Forecast_label_array= eval(vgg19,loss_func,test_dl)
    print("loss: ",test_loss," accuracy: ",test_accuracy)
    for i in range(len(Forecast_label_array)):
        labels = label_list[Forecast_label_array[i]]
        Forecast_label_list.append(labels)

    dataframe = pd.DataFrame({'id':id_array,'label':Forecast_label_list})
    dataframe.to_csv("test.csv",sep=',')