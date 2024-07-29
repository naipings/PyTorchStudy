# 类激活图
# ————2.数据集分析
'''
Malaria Cell Images Dataset 是一个常用的数据集，用于训练和评估计算机视觉模型在疟疾细胞图像分类任务上的性能。
该数据集包含了被感染和未被感染的红血细胞图像，通常用于研究和开发自动检测和分类疟疾细胞的算法。
通过使用这个数据集，可以构建深度学习模型来自动识别感染疟疾的红血细胞，从而帮助医生进行准确的诊断，
'''

# ————3.使用 PyTorch 生成 CAM

# 接下来，我们使用 PyTorch 实现 CAM 生成策略，以了解 CNN 模型能够预测图像可能出现疟疾事件的原因。

# (1)下载数据集，并导入相关库：
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
from random import randint
import cv2
from pathlib import Path
import torch.nn as nn
from torch import optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# (2)指定与输出类别对应的索引：
id2int = {'Parasitized': 0, 'Uninfected': 1}

# (3)执行图像转换操作：
from torchvision import transforms as T

trn_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ColorJitter(brightness=(0.95,1.05), 
                  contrast=(0.95,1.05), 
                  saturation=(0.95,1.05), 
                  hue=0.05),
    T.RandomAffine(5, translate=(0.01,0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]),
])
''' 在以上代码中，对输入图像进行了一系列转换————
首先将图像尺寸调整为 128(最小边为 128), 然后从图像中心进行裁剪。
此外，我们还进行了随机颜色抖动和仿射变换，并使用 .ToTensor() 方法对图像进行缩放(使像素值位于0到1之间), 最后对图像进行归一化处理。
'''

# 对验证集图像执行转换：
val_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]),
])

# (4)定义数据集类 MalariaImages：
class MalariaImages(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        fpath = self.files[ix]
        clss = os.path.basename(Path(fpath).parent)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, clss

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        _imgs, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in _imgs]
        classes = [torch.tensor([id2int[clss]]) for clss in classes]
        imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
        return imgs, classes, _imgs

# (5)获取训练、验证数据集和数据加载器：
all_files = glob('../pytorchStudy/test20/cell_images/*/*.png')
np.random.shuffle(all_files)

from sklearn.model_selection import train_test_split
trn_files, val_files = train_test_split(all_files, random_state=1)

trn_ds = MalariaImages(trn_files, transform=trn_tfms)
val_ds = MalariaImages(val_files, transform=val_tfms)
trn_dl = DataLoader(trn_ds, 32, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, 32, shuffle=False, collate_fn=val_ds.collate_fn)

# (6)定义模型 MalariaClassifier：
def convBlock(ni, no):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(ni, no, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no), # 在卷积神经网络的卷积层之后总会添加 BatchNorm2d 进行数据的归一化处理
        nn.MaxPool2d(2),
    )
    
class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 256),
            convBlock(256, 512),
            convBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(id2int))
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return loss, acc

# (7)定义使用批数据对模型进行训练和验证的函数：
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, labels, _ = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, labels, _ = data
    _preds = model(ims)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()

# (8)训练模型：
model = MalariaClassifier().to(device)
criterion = model.compute_metrics
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5

from torchsummary import summary
print(summary(model, input_size=(3,128,128)))

for ex in range(n_epochs):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        train_loss.append(loss)
        train_acc.append(acc)
    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        val_loss.append(loss)
        val_acc.append(acc)
    avg_train_loss = np.average(train_loss)
    avg_train_acc = np.average(train_acc)
    avg_val_loss = np.average(val_loss)
    avg_val_acc = np.average(val_acc)
    print(f"EPOCH: {ex}	trn_loss: {avg_train_loss}	trn_acc: {avg_train_acc}	val_loss: {avg_val_loss}	val_acc: {avg_val_acc}")

# (9)获取模型中第五个 convBlock 中的卷积层：
im2fmap = nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children())))
''' 在以上代码中，获取模型的第四层以及 convBlock 中的前两层(均为 Conv2D 层)。 '''

# (10)定义 im2gradCAM 函数，该函数接受输入图像并获取与图像激活对应的热力图：
def im2gradCAM(x):
    model.eval()
    logits = model(x)
    heatmaps = []
    activations = im2fmap(x)
    # print(activations.shape)
    pred = logits.max(-1)[-1]
    # 获取模型预测
    model.zero_grad()
    # 计算相对于模型置信度最高的 logits 的梯度
    logits[0,pred].backward(retain_graph=True)
    # 获取所需特征图位置的梯度，并对每个特征图取平均梯度
    pooled_grads = model.model[-6][1].weight.grad.data.mean((1,2,3))
    # 将每个激活图与对应的梯度平均值相乘
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i]
    # 计算所有加权激活图的平均值
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'

# (11)定义 upsampleHeatmap 函数将热图上采样为与图像形状对应的形状：
SZ = 120
def upsampleHeatmap(map, img):
    m,M = map.min(), map.max()
    map = 255 * ((map-m) / (M-m))
    map = np.uint8(map)
    map = cv2.resize(map, (SZ,SZ))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map*0.7 + img*0.3)
    return map
''' 在前面的代码行中，我们对图像进行了反归一化，并将热图覆盖在图像之上。 '''

# (12)使用一组测试图像调用上述函数：
N = 20
_val_dl = DataLoader(val_ds, batch_size=N, shuffle=True, collate_fn=val_ds.collate_fn)
x,y,z = next(iter(_val_dl))

for i in range(N):
    image = cv2.resize(z[i], (SZ, SZ))
    heatmap, pred = im2gradCAM(x[i:i+1])
    if(pred=='Uninfected'):
        continue
    heatmap = upsampleHeatmap(heatmap, image)
    plt.figure(figsize=(5,3))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(heatmap)
    plt.suptitle(pred)
    plt.show()

'''根据结果图可以看出，
预测结果是由红色高亮区域的内容决定的(这部分区域具有最高的 CAM 值)。
学习了如何使用训练好的模型生成图像的类激活热力图，我们就可以解释是什么原因导致了模型产生某个分类结果。
'''