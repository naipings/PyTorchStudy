# 可视化学习特征的结果

'''
可视化特征学习的结果具有多方面的应用。
首先，可视化可以帮助我们评估和调整神经网络的设计，
通过观察特征图和梯度分布，可以判断网络是否学到了有效的特征表示，从而优化网络结构和参数设置；

其次，可视化还可以帮助我们解释网络的预测结果，
通过观察中间层的输出，我们可以了解网络对不同类别或输入样本的响应模式，解释其预测的依据；

最后，通过观察网络学到的特征表示，可以借鉴其中的思想，设计更好的手工特征或特征提取算法。
'''

# 在本节中，我们将探索神经网络究竟学到了什么，使用卷积神经网络对包含 X 和 O 的图像的数据集进行分类，并检查网络层输出了解激活结果。

# (1)首先下载数据集并解压

# (2)导入所需库
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np, cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
from imgaug import augmenters as iaa

# (3)定义获取数据的类，确保图像形状已调整为 28*28，并且目标类别转换为数值形式。
# 定义图像增强方法，将图像调整为 28*28：
tfm = iaa.Sequential(iaa.Resize(28))
# 定义一个将文件夹路径作为输入的类，并在 __init__ 方法中遍历该路径中的文件：
class XO(Dataset):
    def __init__(self, folder):
        self.files = glob(folder)
    
    # 定义 __len__ 方法返回数据集的长度：
    def __len__(self):
        return len(self.files)
    
    # 定义 __getitem__ 方法获取索引，返回该索引处存在的文件，读取图像文件并对图像执行增强。
    # 这里未使用 collate_fn，因为小数据集不会显著影响训练时间：
    def __getitem__(self, ix):
        f = self.files[ix]
        im = tfm.augment_image(cv2.imread(f)[:,:,0])

        # 在图像形状(每个图像的形状为 28*28)前创建通道尺寸：
        im = im[None]

        # 根据文件名中的字符 “/” 和 “@” 之间的字符确定每个图像的类别：
        cl = f.split('/')[-1].split('@')[0] == 'x'

        # 最后，返回图像及其对应类别：
        return torch.tensor(1 - im/255).to(device).float(), torch.tensor([cl]).float().to(device)

# (4)显示图像样本，通过上述定义的类提取图像及其对应的类：
data = XO('../pytorchStudy/test19/images/*')

# 根据获得的数据集绘制图像样本：
R, C = 7,7
fig, ax = plt.subplots(R, C, figsize=(5,5))
for label_class, plot_row in enumerate(ax):
    for plot_cell in plot_row:
        plot_cell.grid(False); plot_cell.axis('off')
        ix = np.random.choice(1000)
        im, label = data[ix]
        plot_cell.imshow(im[0].cpu(), cmap='gray')
plt.tight_layout()
plt.show()


# (5)定义模型架构、损失函数和优化器：
from torch.optim import SGD, Adam
def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

# 由于是二分类问题，此处使用二元交叉熵损失(nn.BCELoss())，打印模型摘要：
from torchsummary import summary
model, loss_fn, optimizer = get_model()
print(summary(model, input_size=(1,28,28)))
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 26, 26]             640
         MaxPool2d-2           [-1, 64, 13, 13]               0
              ReLU-3           [-1, 64, 13, 13]               0
            Conv2d-4          [-1, 128, 11, 11]          73,856
         MaxPool2d-5            [-1, 128, 5, 5]               0
              ReLU-6            [-1, 128, 5, 5]               0
           Flatten-7                 [-1, 3200]               0
            Linear-8                  [-1, 256]         819,456
              ReLU-9                  [-1, 256]               0
           Linear-10                    [-1, 1]             257
          Sigmoid-11                    [-1, 1]               0
================================================================
Total params: 894,209
Trainable params: 894,209
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.69
Params size (MB): 3.41
Estimated Total Size (MB): 4.10
----------------------------------------------------------------
"""

# (6)定义用于批训练的函数，该函数使用图像及其分类作为输入，并在对给定的批数据上执行反向传播后返回其损失值和准确率：
def train_batch(x, y, model, optimizer, loss_fn):
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

def accuracy(x, y, model):
    with torch.no_grad():
        prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()

# (7)定义 DataLoader，其中输入是 Dataset 类：
trn_dl = DataLoader(data, batch_size=32, drop_last=True)
val_dl = DataLoader(data, batch_size=len(data))

# (8)初始化并训练模型：
model, loss_fn, optimizer = get_model()

for epoch in range(10):
    print(epoch+1)
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)

# (9)获取图像以查看滤波器学习到的图像内容：
# im, c = trn_dl.dataset[2]
# plt.imshow(im[0].cpu())
# plt.show()

# 自我修改：
for i in range(10):
    im, c = trn_dl.dataset[i]
    plt.imshow(im[0].cpu())
    plt.show()

# ----------------------------------------------------------------
# train_losses, train_accuracies = [], []
# val_losses, val_accuracies = [], []

# # 循环 10 个 epoch 并初始化包含给定 epoch 内各批训练数据的准确率和损失的列表：
# for epoch in range(10):
#     print(epoch+1)
#     train_epoch_losses, train_epoch_accuracies = [], []

#     # 遍历一批训练数据并计算一个 epoch 内的损失值(train_epoch_loss)和准确率(train_epoch_accuracy)：
#     for ix, batch in enumerate(iter(trn_dl)):
#         x, y = batch
#         batch_loss = train_batch(x, y, model, optimizer, loss_fn)
#         train_epoch_losses.append(batch_loss) 
#     train_epoch_loss = np.array(train_epoch_losses).mean()

#     for ix, batch in enumerate(iter(trn_dl)):
#         x, y = batch
#         is_correct = accuracy(x, y, model)
#         train_epoch_accuracies.extend(is_correct)
#     train_epoch_accuracy = np.mean(train_epoch_accuracies)

#     # 计算验证数据的损失值和准确率(验证数据的批大小等于验证数据的长度)
#     for ix, batch in enumerate(iter(val_dl)):
#         x, y = batch
#         val_is_correct = accuracy(x, y, model)
#         validation_loss = val_loss(x, y, model, loss_fn)
#     val_epoch_accuracy = np.mean(val_is_correct)

#     # 最后，将训练和验证数据集的损失值和准确率添加到相应列表中，以查看模型训练的改进：
#     train_losses.append(train_epoch_loss)
#     train_accuracies.append(train_epoch_accuracy)
#     val_losses.append(validation_loss)
#     val_accuracies.append(val_epoch_accuracy)

# # (9)可视化模型训练过程中训练和验证数据集中的损失值和准确率的变化：
# epochs = np.arange(10)+1
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# plt.subplot(121)
# plt.plot(epochs, train_losses, 'bo', label='Training loss')
# plt.plot(epochs, val_losses, 'r', label='Validation loss')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation loss when batch size is 32')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid('off')
# plt.subplot(122)
# plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation accuracy when batch size is 32')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
# plt.gca().yaxis.set_major_locator(mticker.FixedLocator(plt.gca().get_yticks()))
# ticks_loc = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in ticks_loc])
# plt.legend()
# plt.grid('off')
# plt.show()
# ----------------------------------------------------------------