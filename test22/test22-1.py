# 面部关键点检测

# 接下来，使用 PyTorch 实现面部关键点检测模型：

# (1)导入相关库：
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np, pandas as pd, os, glob
import matplotlib
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
import glob
import cv2
import copy
from copy import deepcopy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# (2)下载并导入相关数据，下载包含图像及其对应的面部关键点的数据集：
root_dir = '../pytorchStudy/test22/P1_Facial_Keypoints/data/training/'
all_img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
data = pd.read_csv('../pytorchStudy/test22/P1_Facial_Keypoints/data/training_frames_keypoints.csv')

# (3)定义为数据加载器提供输入和输出样本的 FacesData 类：
class FacesData(Dataset):
    # 定义 __init__ 方法，以二维数据表格(df)作为输入：
    def __init__(self, df):
        super(FacesData).__init__()
        self.df = df
        
        # 定义用于预处理图像的均值和标准差，供预训练 VGG16 模型使用：
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    # 定义 __len__ 方法：
    def __len__(self):
        return len(self.df)

    # 定义 __getitem__ 方法，获取与给定索引对应的图像，对其进行缩放，
    # 获取与给定索引对应的关键点值，对关键点进行归一化，以便我们获取关键点的相对位置，并对图像进行预处理。
    
    # 定义 __getitem__ 方法并获取与给定索引(ix)对应的图像路径：
    def __getitem__(self, ix):
        img_path = '../pytorchStudy/test22/P1_Facial_Keypoints/data/training/' + self.df.iloc[ix,0]
        ''' 因为在 .scv 文件中, 第1列是图片名称, 后面都是图片上的关键点, 所以使用 self.df.iloc[ix,0] 获取第一列的图片名称。 '''

        # 缩放图像：
        img = cv2.imread(img_path)/255.

        # 将目标输出值(关键点位置)根据原始图像尺寸的比例进行归一化：
        kp = deepcopy(self.df.iloc[ix,1:].tolist())
        kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()
        ''' 在以上代码中，确保关键点按原始图像尺寸的比例计算，这样做是为了当我们调整原始图像的尺寸时，关键点的位置不会改变。 '''

        # 对图像进行预处理后返回关键点(kp2) 和图像(img)：
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2) 
        img = self.preprocess_input(img)
        return img, kp2

    # 定义预处理图像函数(preprocess_input)：
    def preprocess_input(self, img):
        img = cv2.resize(img, (224,224))
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img.to(device)

    # 定义函数加载图像，用于可视化测试图像和测试图像的预测关键点：
    def load_img(self, ix):
        img_path = '../pytorchStudy/test22/P1_Facial_Keypoints/data/training/' + self.df.iloc[ix,0]        
        img = cv2.imread(img_path)
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (224,224))
        return img

# (4)拆分训练和测试数据集，并构建训练和测试数据集和数据加载器：
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=101)
train_dataset = FacesData(train.reset_index(drop=True))
test_dataset = FacesData(test.reset_index(drop=True))

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# (5)定义用于识别图像中关键点的模型。
# 加载预训练的 VGG16 模型：
def get_model():
    model = models.vgg16(pretrained=True)

    # 冻结预训练模型参数：
    for param in model.parameters():
        param.requires_grad = False

    # 重建模型最后两层并训练参数：
    model.avgpool = nn.Sequential(nn.Conv2d(512,512,3),
                nn.MaxPool2d(2),
                nn.Flatten())
    model.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 136),
                nn.Sigmoid())
    ''' 分类器模块的最后一层使用 Sigmoid 函数，它返回介于 0 和 1 之间的值，因为关键点位置是相对于原始图像尺寸的相对位置，因此预期输出将始终介于 0 和 1 之间。 '''

    # 定义损失函数(使用平均绝对误差)和优化器：
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), criterion, optimizer

# (6)初始化模型，损失函数，以及对应的优化器：
model, criterion, optimizer = get_model()

# (7)定义函数在训练数据集上进行训练并在测试数据集上进行验证。
# train_batch() 函数根据输入计算模型输出、损失值并执行反向传播以更新权重：
def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

# 构建函数返回测试数据集的损失和预测的关键点：
@torch.no_grad()
def validate_batch(img, kps, model, criterion):
    model.eval()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss

# (8)训练模型，并在测试数据上对其进行测试：
train_loss, test_loss = [], []
n_epochs = 50

for epoch in range(n_epochs):
    print(f" epoch {epoch+ 1}/50")
    epoch_train_loss, epoch_test_loss = 0, 0
    for ix, (img,kps) in enumerate(train_loader):
        loss = train_batch(img, kps, model, optimizer, criterion)
        epoch_train_loss += loss.item() 
    epoch_train_loss /= (ix+1)

    for ix,(img,kps) in enumerate(test_loader):
        ps, loss = validate_batch(img, kps, model, criterion)
        epoch_test_loss += loss.item() 
    epoch_test_loss /= (ix+1)

    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)

# (9)绘制模型训练过程中的损失和测试损失：
epochs = np.arange(50)+1
import matplotlib.pyplot as plt
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, test_loss, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

# (10)使用随机测试图像来测试模型，利用 FacesData 类中的 load_img 方法：
ix = 20
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.title('Original image')
im = test_dataset.load_img(ix)
plt.imshow(im)
plt.grid(False)
plt.subplot(122)
plt.title('Image with facial keypoints')
x, _ = test_dataset[ix]
plt.imshow(im)
kp = model(x[None]).flatten().detach().cpu()
plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
plt.grid(False)
plt.show()

# 从结果图可知，给定输入图像，模型能够相当精准地识别面部关键点。