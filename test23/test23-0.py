# 多任务学习
# ————实现年龄估计和性别分类

# 接下来，使用 PyTorch 实现多任务学习模型：

# (1)导入相关库：
import torch
import numpy as np, cv2, pandas as pd, time, os
import matplotlib
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# (2)加载下载完成的数据集，查看数据结构：
trn_df = pd.read_csv('../pytorchStudy/test23/Multi_Task_Learning/fairface-labels-train.csv')
val_df = pd.read_csv('../pytorchStudy/test23/Multi_Task_Learning/fairface-labels-val.csv')
print(trn_df.head())
'''打印结果如下：
          file  age  gender        race  service_test
0  train/1.jpg   59    Male  East Asian          True
1  train/2.jpg   39  Female      Indian         False
2  train/3.jpg   11  Female       Black         False
3  train/4.jpg   26  Female      Indian          True
4  train/5.jpg   26  Female      Indian          True
'''



# 引入数据增强管道：
from imgaug import augmenters as iaa
aug = iaa.Sequential([
            iaa.Multiply(0.5),
            iaa.LinearContrast(0.5)], random_order= True)

def to_numpy(tensor):
    return tensor.cpu().numpy()



# (3)构建 GenderAgeClass 类，以文件名作为输入并返回相应的图像、性别和年龄。
# 年龄需要缩放，因为这是一个连续的数字，缩放数据以避免梯度损失，然后在后处理期间重新进行还原。

# 在 __init__ 方法中以图像的文件路径作为输入：
IMAGE_SIZE = 224
class GenderAgeClass(Dataset):
    def __init__(self, df, tfms=None, aug=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self.aug=aug
    # 使用 __len__ 方法返回输入中图像的数量：
    def __len__(self):
        return len(self.df)

    # 定义 __getitem__ 方法，获取给定位置 ix 的图像信息：
    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = '../pytorchStudy/test23/Multi_Task_Learning/' + f.file
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen

    # 编写图像预处理函数，包括调整图像大小、调整图像通道以及图像归一化：
    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im/255.)
        return im[None]

    # 创建 collate_fn 方法，该方法用于对批数据执行以下预处理：
    '''
        使用 preprocess_image 方法处理每个图像。
        缩放年龄(除以数据集中存在的最大年龄值————80), 令所有值都介于 0 和 1 之间。
        将性别转换为浮点值。
        将图像、年龄和性别转换为张量对象并返回。
    '''
    def collate_fn(self, batch):
        'preprocess images, ages and genders'
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)

            ages.append(float(int(age)/80))
            genders.append(float(gender))

        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
        ims = torch.cat(ims).to(device)

        if self.aug:
            ims=self.aug.augment_images(images=list(map(to_numpy,list(ims))))
        ims = torch.tensor(np.array(ims))[:,:,:].to(device)

        return ims, ages, genders


# (4)定义训练和验证数据集以及数据加载器。
# 创建数据集：
trn = GenderAgeClass(trn_df,aug=aug)
val = GenderAgeClass(val_df,aug=aug)
# 构建数据加载器：
train_loader = DataLoader(trn, batch_size=32, shuffle=True, drop_last=True, collate_fn=trn.collate_fn)
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn)
a,b,c, = next(iter(train_loader))
print(a.shape, b.shape, c.shape)
# torch.Size([32, 3, 224, 224]) torch.Size([32]) torch.Size([32])


# (5)定义模型、损失函数和优化器。
# 在函数 get_model() 中，加载预训练 VGG16 模型：
def get_model():
    model = models.vgg16(pretrained = True)

    # 冻结加载的模型(指定参数 param.requires_grad = False)：
    for param in model.parameters():
        param.requires_grad = False

    # 使用自定义网络层替换 avgpool 层：
    model.avgpool = nn.Sequential(
        nn.Conv2d(512,512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )

    # 构建名为 ageGenderClassifier 的神经网络类，以创建包含两个输出分支的神经网络：
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()

            # 定义中间层 intermediate：
            self.intermediate = nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128,64),
                nn.ReLU(),
            )

            # 定义 age_classifier 和 gender_classifier：
            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            '''
            在以上代码中，年龄预测层 age_classifier 和性别预测层 gender_classifier 均使用 Sigmoid 激活，
            因为年龄输出是一个介于 0 和 1 之间的值，且性别输出是 0 或 1 。
            '''

        # 定义前向传递方法 forward, 使用网络层 age_classifier 和 gender_classifier：
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age

    # 使用自定义网络替换 VGG16 预训练模型的分类器模块：
    model.classifier = ageGenderClassifier()

    # 定义性别分类(二元交叉熵损失)和年龄预测(L1损失)的损失函数。定义优化器并返回模型、损失函数和优化器：
    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    loss_functions = gender_criterion, age_criterion
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    
    return model.to(device), loss_functions, optimizer

# 调用 get_model() 函数初始化变量中的值：
model, loss_functions, optimizer = get_model()


# (6)定义函数在训练数据集上进行训练并在测试数据集上进行验证。
# train_batch 方法将图像、性别、年龄、模型、优化器和损失函数的实际值作为输入来计算总损失。
# 使用适当的u输入参数定义 train_batch() 方法：
def train_batch(data, model, optimizer, criteria):
    # 指定训练模型，将优化器重置为 zero_grad, 并计算年龄和性别的预测值：
    model.train()
    ims, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(ims) 

    # 在计算年龄估计和性别分类对应的损失之前，获取用于年龄估计和性别分类的损失函数：
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)

    # 通过将 gender_loss 和 age_loss 相加来计算整体损失，并通过优化模型的可训练权重执行反向传播以减少整体损失：
    total_loss = gender_loss + age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.cpu().detach().numpy()

# validate_batch() 方法将图像、模型和损失函数以及年龄和性别的实际值作为输入，计算年龄和性别的预测值以及损失值。
# 使用所需的输入参数定义 validate_batch 函数：
def validate_batch(data, model, criteria):
    # 指定模型处于评估阶段，因此不需要进行梯度计算：
    model.eval()
    ims, age, gender = data
    with torch.no_grad():
        pred_gender, pred_age = model(ims)

    # 计算年龄和性别预测对应的损失值(gender_loss 和 age_loss)。压缩预测形状(batch size, 1)，以便将其整形为与目标值相同的形状(batch size)：
    gender_criterion, age_criterion = criteria
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)

    # 计算整体损失，最终预测的性别类别(pred_gender)、性别预测准确率和年龄估计误差：
    total_loss = gender_loss + age_loss
    pred_gender = (pred_gender > 0.5).squeeze()
    gender_acc = (pred_gender == gender).float().sum()
    age_mae = torch.abs(age - pred_age).float().sum()
    return total_loss.cpu().detach().numpy(), gender_acc.cpu().detach().numpy(), age_mae.cpu().detach().numpy()



from torch import optim
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                factor=0.5,
                                                patience=0,
                                                threshold = 0.0001,
                                                verbose=True,
                                                min_lr = 1e-5,
                                                threshold_mode = 'abs')



# (7)训练模型。
# 定义用于存储训练和测试损失值的列表，并指定训练 epoch 数：
model, criteria, optimizer = get_model()
val_gender_accuracies = []
val_age_maes = []
train_losses = []
val_losses = []

n_epochs = 10
best_test_loss = 1000
start = time.time()

# 在每个 epoch 开始时重新初始化训练和测试损失值：
for epoch in range(n_epochs):
    print(f" epoch {epoch+ 1}/{n_epochs}")
    epoch_train_loss, epoch_test_loss = 0, 0
    val_age_mae, val_gender_acc, ctr = 0, 0, 0
    _n = len(train_loader)

    # 遍历训练数据加载器(train_loader)并训练模型：
    for ix, data in enumerate(train_loader):
        # if ix == 100: break
        loss = train_batch(data, model, optimizer, criteria)
        epoch_train_loss += loss.item()

    # 遍历测试数据加载器并计算性别及年龄预测准确率：
    for ix, data in enumerate(test_loader):
        # if ix == 10: break
        loss, gender_acc, age_mae = validate_batch(data, model, criteria)
        epoch_test_loss += loss.item()
        val_age_mae += age_mae
        val_gender_acc += gender_acc
        ctr += len(data[0])

    # 计算年龄预测和性别分类的整体准确率：
    val_age_mae /= ctr
    val_gender_acc /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(test_loader)
    scheduler.step(epoch_test_loss)

    # 打印每个 epoch 结束时模型性能指标：
    elapsed = time.time()-start
    best_test_loss = min(best_test_loss, epoch_test_loss)
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start, (n_epochs-epoch)*(elapsed/(epoch+1))))
    info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}\tBest Test Loss: {best_test_loss:.4f}'''
    info += f'\nGender Accuracy: {val_gender_acc*100:.2f}%\tAge MAE: {val_age_mae:.2f}\n'
    print(info)

    # 存储每个 epoch 中测试数据集的年龄和性别预测准确率：
    val_gender_accuracies.append(val_gender_acc)
    val_age_maes.append(val_age_mae)


# (8)绘制年龄估计和性别预测训练过程的准确率变化：
epochs = np.arange(1,len(val_gender_accuracies)+1)
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flat
ax[0].plot(epochs, val_gender_accuracies, 'bo')
ax[1].plot(epochs, val_age_maes, 'r')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('MAE')
ax[0].set_title('Validation Gender Accuracy')
ax[1].set_title('Validation Age Mean-Absolute-Error')
plt.show()
'''
在年龄预测方面平均与真实年龄相差了 6 岁左右，在性别预测方面的准确率约为 84% 。
'''

torch.save(model.state_dict(), "../pytorchStudy/test23/myvgg19_model_genderAge_2.pth")

# (9)随机选择测试图像，预测图像中人物的年龄和性别。
# 获取并加载图像，将其输入到 trn 对象中的 preprocess_image 方法中：
im = cv2.imread('../pytorchStudy/test23/23-1imgs/test2.jpg')
im = trn.preprocess_image(im).to(device)

# 通过训练好的模型传递图像：
gender, age = model(im)
pred_gender = gender.to('cpu').detach().numpy()
pred_age = age.to('cpu').detach().numpy()

# 绘制图像并打印真实值和预测值：
im = cv2.imread('../pytorchStudy/test23/23-1imgs/test2.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
print('predicted gender:',np.where(pred_gender[0][0]<0.5,'Male','Female'), '; Predicted age', int(pred_age[0][0]*80))
# predicted gender: Female ; Predicted age 28

"""
综上，可以看到，我们能够一次性同时预测年龄和性别。
但是，需要注意，本节构建的模型非常不稳定，年龄值会随着图像的光照条件有很大差异，可以通过使用数据增强观察模型性能改善情况。
"""


im = cv2.imread('../pytorchStudy/test23/23-1imgs/test7.jpg')
im = trn.preprocess_image(im).to(device)
# 通过训练好的模型传递图像：
gender, age = model(im)
pred_gender = gender.to('cpu').detach().numpy()
pred_age = age.to('cpu').detach().numpy()
# 绘制图像并打印真实值和预测值：
im = cv2.imread('../pytorchStudy/test23/23-1imgs/test7.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
print('predicted gender:',np.where(pred_gender[0][0]<0.5,'Male','Female'), '; Predicted age', int(pred_age[0][0]*80))


im = cv2.imread('../pytorchStudy/test23/23-1imgs/test8.jpg')
im = trn.preprocess_image(im).to(device)
# 通过训练好的模型传递图像：
gender, age = model(im)
pred_gender = gender.to('cpu').detach().numpy()
pred_age = age.to('cpu').detach().numpy()
# 绘制图像并打印真实值和预测值：
im = cv2.imread('../pytorchStudy/test23/23-1imgs/test8.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
print('predicted gender:',np.where(pred_gender[0][0]<0.5,'Male','Female'), '; Predicted age', int(pred_age[0][0]*80))


im = cv2.imread('../pytorchStudy/test23/23-1imgs/test9.jpg')
im = trn.preprocess_image(im).to(device)
# 通过训练好的模型传递图像：
gender, age = model(im)
pred_gender = gender.to('cpu').detach().numpy()
pred_age = age.to('cpu').detach().numpy()
# 绘制图像并打印真实值和预测值：
im = cv2.imread('../pytorchStudy/test23/23-1imgs/test9.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
print('predicted gender:',np.where(pred_gender[0][0]<0.5,'Male','Female'), '; Predicted age', int(pred_age[0][0]*80))
