# 迁移学习
# ————3.使用预训练 VGG16 模型实现猫狗分类
''' 接下来，我们介绍如何在实践中使用 VGG16 模型，仅使用 1000 张图像(猫、狗图像各 500 张)构建猫狗分类模型。 '''

# (1)导入所需要的库：
import torch.nn as nn
import torch
from torchvision import transforms,models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2, glob, numpy as np
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# (2)下载猫狗分类数据集、解压并指定训练和测试目录：
train_data_dir = '../pytorchStudy/test21/archive/training_set/training_set'
test_data_dir = '../pytorchStudy/test21/archive/test_set/test_set'

# (3)定义 Dataset 类，用于返回猫狗数据集的输入-输出对，使用 transforms 模块中的 Normalize 调用 normalize 函数执行数据归一化，并获取每个文件夹中的前 500 张图像：
class CatsDogs(Dataset):
    def __init__(self, folder):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        from random import shuffle, seed; seed(10); shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths] 
    def __len__(self):
        return len(self.fpaths)
    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224,224))
        im = torch.tensor(im/255)
        im = im.permute(2,0,1)
        im = self.normalize(im) 
        return im.float().to(device), torch.tensor([target]).float().to(device)
'''

'''

# (3)获取图像及其标签，抽样检查示例图像及其对应的类别标签：
data = CatsDogs(train_data_dir)

im, label = data[100]
plt.imshow(im.permute(1,2,0).cpu())
plt.show()
print(label)
# tensor([0.], device='cuda:0')

# (4)定义模型，下载预训练的 VGG16 权重，冻结特征提取模块，并训练新的分类器。
# 首先，利用 models 类下载预训练的 VGG16 模型：
def get_model():
    model = models.vgg16(pretrained=True)
    # model = models.vgg19(pretrained=True)
    # model = models.vgg11(pretrained=True)

    # 通过指定 param.requires_grad = False 在反向传播期间冻结模型参数：
    for param in model.parameters():
        param.requires_grad = False

    # 替换 avgpool 模块以返回大小为 1*1 的特征图替换原模型中 7*7 的特征图，即输出尺寸变为 batch_size * 512 * 1 * 1：
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    '''  '''

    # 定义模型的分类器模块，首先展平 avgpool 模块的输出，然后连接到具有 128 个单元的隐藏层，并在执行激活函数后连接到输出层：
    model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
    )

    # 定义损失函数(loss_fn)、优化器：
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer


"""
在以上代码中，我们首先冻结了预训练模型的所有参数，然后定义了新的 avgpool 和分类器模块，打印模型摘要：
"""
from torchsummary import summary
model, criterion, optimizer = get_model()
print(summary(model, (3,224,224)))
''' 打印结果见txt文件, 
输出中省去了与预训练模型中相同的网络层，模型共有 1470 万个参数，其中可训练参数的数量只有 65793 个，因为模型中只有分类器模块具有要学习的权重。
'''

# (5)定义函数获取数据、训练模型、计算准确度：
def get_data():
    train = CatsDogs(train_data_dir)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last = True)
    val = CatsDogs(test_data_dir)
    val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last = True)
    return trn_dl, val_dl

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
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()

# (6)初始化 get_data() 和 get_model() 函数：
trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

# (7)训练模型，并绘制模型训练过程中训练和测试准确率：
train_losses, train_accuracies = [], []
val_accuracies = []
for epoch in range(10):
    print(f" epoch {epoch + 1}/10")
    train_epoch_losses, train_epoch_accuracies = [], []
    val_epoch_accuracies = []

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss) 
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        val_epoch_accuracies.extend(val_is_correct)
    val_epoch_accuracy = np.mean(val_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_accuracies.append(val_epoch_accuracy)

epochs = np.arange(10)+1
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy with VGG19 \nand 1K training data points')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.95,1)
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().yaxis.set_major_locator(mticker.FixedLocator(plt.gca().get_yticks()))
ticks_loc = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in ticks_loc])
plt.legend()
plt.grid('off')
plt.show()

'''根据结果图可知，
可以看到，在第一个 epoch 结束时模型准确率就可以达到 98%, 即使训练数据集仅包含 1000 张图像(每个类别 500 张图像)。
除了 VGG16, 也可以使用 VGG11 和 VGG19 预训练神经网络架构，它们的工作方式与 VGG16 类似, 仅层数不同。VGG19 比 VGG16 有更多的参数，因为它具有更多的网络层。
'''