# 迁移学习
# ————2.VGG16 架构
'''
Visual Geometry Group (VGG) 是由牛津大学的研究人员在 2014 年提出的卷积神经网络架构，
其中 16 代表模型中的层数，包含 13 层的卷积层, 3 层的全连接层。 
VGG16 分类模型在 2014 年的 ImageNet 比赛中获得了亚军，是一种广泛应用于计算机视觉任务的模型架构。
VGG16 中的卷积层使用小尺寸的卷积核，以增加网络深度，提高模型的非线性能力，并且能够提取更丰富的特征。
接下来，我们介绍 VGG16 架构以及如何在 PyTorch 中使用 VGG16 预训练模型。
'''

# (1)导入所需要的库：
import torch
from torchvision import transforms,models,datasets
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'
''' torchvision 包中的 models 模块提供了 PyTorch 中可用的预训练模型。 '''

# (2)加载 VGG16 模型并在设备内注册模型：
model = models.vgg16(pretrained=True).to(device)
''' 在以上代码中, model 类中调用了 vgg16 方法，通过使用 pretrained=True 指定加载用于在 ImageNet 竞赛中对图像进行分类的权重，然后将模型注册到设备中。 '''
# 注：模型本地下载位置：/home/lcz/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# (3)打印模型摘要：
print(summary(model, (3, 224, 224)))
''' 模型摘要输出见txt文件 '''

# 该网络中约有 1.38 亿个参数(网络末端的线性层包括约 102万 + 16万 + 400万 = 1.22亿个参数)，其中包括 13 个卷积层和 3 个线性层，可以使用 models 打印 VGG16 模型组件：
print(model)
''' 输出见txt文件, 
模型中包括三个主要的子模块————features、avgpool 和 classifier, 通常，需要冻结 features 和 avgpool 模块，
删除分类器(classifier)模块并在其位置创建一个新的 classifier 模块，用于预测新任务所需的图像类别。
'''