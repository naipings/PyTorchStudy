# 可视化不同网络层的特征图

''' 19-1代码 '''
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



''' 本节内容代码 '''
# (1)提取原始 O 图像从输入层到第 2 个卷积层的输出，绘制第 2 层中滤波器与输入 O 图像进行卷积后的输出。
# 绘制滤波器与相应图像的卷积输出：
second_layer = nn.Sequential(*list(model.children())[:4])
second_intermediate_output = second_layer(im[None])[0].detach()

print(second_intermediate_output.shape)
# torch.Size([128, 11, 11])
n = 11
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(second_intermediate_output[ix].cpu())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()

print(im.shape)
# torch.Size([1, 28, 28])

# 从数据集中获取多张 O 图像：
x, y = next(iter(trn_dl))
x2 = x[y==0]
print(len(x2))
# 15

# 调整 x2 形状使其能够作为卷积神经网络的输入，即 批大小*通道*高度*宽度 ：
x2 = x2.view(-1,1,28,28)

# 以第 34 个滤波器的输出为例，当我们通过滤波器 34 传递多个 O 图像时，可以在不同图像之间看到相似的激活：
second_layer = nn.Sequential(*list(model.children())[:4])
second_intermediate_output = second_layer(x2).detach()

print(second_intermediate_output.shape)
# torch.Size([15, 128, 11, 11])
n = 4
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    if ix < n**2-1:
        axis.imshow(second_intermediate_output[ix,34,:,:].cpu())
        axis.set_title(str(ix))
plt.tight_layout()
plt.show()

print(len(data))
# 2498


# (2)绘制全连接层的激活。
# 首先，获取图像样本：
custom_dl = DataLoader(data, batch_size=2498, drop_last=True)

# 接下来，从数据集中选择 O 图像，对它们进行整形后，作为输入传递到 CNN 模型中：
x, y = next(iter(custom_dl))
x2 = x[y==0]
print(len(x2))
# 1245
x2 = x2.view(len(x2),1,28,28)

# 将以上图像通过 CNN 模型，获取全连接层输出：
flatten_layer = nn.Sequential(*list(model.children())[:7])
flatten_layer_output = flatten_layer(x2).detach()

print(flatten_layer_output.shape)
# torch.Size([1245, 3200])

# 绘制全连接层输出：
plt.figure(figsize=(100,10))
plt.imshow(flatten_layer_output.cpu())
plt.show()

'''
输出的形状为 1245*3200, 因为数据集中有 1245 个 O 图像，且在全连接层中的每个图像的输出为 3200 维。
当输入为 O 时，全连接层中大于零的激活值会突出显示，在图像中显示为白色像素。
可以看到，模型已经学习了图像中的结构信息，即使属于同一类别的输入图像存在较大的风格差异。
'''