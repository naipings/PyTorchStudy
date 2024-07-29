# 实现 R-CNN 目标检测
'''
我们已经从理论上讲解了 R-CNN 的工作原理，在本节中，我们将学习如何使用 PyTorch 构建 R-CNN 模型。
    1.准备数据集
    2.定义区域提议提取和IoU计算函数
    3.创建训练数据
      为模型创建输入数据
      调整区域提议尺寸
      使用预训练模型提取区域特征图
      为模型创建输出数据
      使用预定义类别或背景标签标记每个区域提议
      如果区域提议对应于目标对象而不非背景，则获取区域提议与真实边界框之间的偏移量
    4.定义并训练神经网络模型
    5.预测新图像
'''

# 1.数据集准备
'''
为了构建目标检测模型，从 Kaggle 中下载公开数据集，为了简单起见，在代码中，我们只处理包含公共汽车或卡车的图像。

可以看到图像及其相应的标签存储在 CSV 文件中：
其中, XMin、XMax、YMin 和 YMax 对应于图像中目标对象的边界框坐标, LabelName 对应于图像类别。

数据集下载完成后，我们继续对数据集进行处理以用于模型训练：
    1.获取每个图像及其对应的类别和边界框。
    2.获取每个图像中的区域提议及其与真实边界框对应的交并比 (Intersection over union, IoU), 
      以及区域提议相对于真实边界框需要进行修正的偏移量。
    3.为每个类别分配数字标签(除了公交车和卡车类别之外，还需要一个额外的背景类别), 
      当区域提议与真实边界框的 IoU 值低于阈值时即被视为背景类别。
    4.将每个区域提议调整为相同大小，以便将它们输入到神经网络。

综上，我们需要调整区域提议的大小，为每个区域提议分配标签，并计算区域提议相对于真实边界框的偏移量。
'''

# (1)指定图像的位置并读取 CSV 文件中的真实边界框数据：
import selectivesearch
from torchvision import transforms, models, datasets
from torchvision.ops import nms
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
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_ROOT = '../pytorchStudy/test25/open-images-bus-trucks/images/images'
DF_RAW = pd.read_csv('../pytorchStudy/test25/open-images-bus-trucks/df.csv')
print(DF_RAW.head())


# (2)定义类 OpenImages，返回图像及其包含的目标对象的类别、目标对象边界框以及图像的文件路径。
'''
将数据帧 (df) 和图像文件夹路径 (image_folder) 作为输入传递给 __init__ 方法，
并提取数据帧中不重复的 ImageID 值 (self.unique_images), 
这是因为一张图像可能包含多个对象，因此数据帧中多个行可能对应于相同的 ImageID 值：
'''
class OpenImages(Dataset):
    def __init__(self, df, image_folder=IMAGE_ROOT):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()
    def __len__(self):
        return len(self.unique_images)

    # 定义 __getitem__ 方法，获取与索引(ix) 对应的图像(image_id)，图像中目标对象的边界框坐标(box)、类别，
    # 并返回图像、边界框、类别和图像路径：
    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)[...,::-1] # conver BGR to RGB
        h, w, _ = image.shape
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes * np.array([w,h,w,h])).astype(np.uint16).tolist()
        classes = df['LabelName'].values.tolist()
        return image, boxes, classes, image_path


# (3)检查样本图像及图像中包含的目标对象的类别和边界框：
ds = OpenImages(df=DF_RAW)
im, bbs, clss, _ = ds[21]

# 自我补充(为了yolo4训练，保存图像数据为txt文件）：
# 需要numpy==1.22.3，但是本人下载后说这个版本与tensorflow不匹配，所以本人执行完这部分代码后，还是下载回了1.24.3)
# for i in range(len(ds)):
#     im, bbs, clss, _ = ds[i]
#     height, width = im.shape[:2]

#     full_name = os.path.basename(_)
#     file_name_without_extension = os.path.splitext(full_name)[0]
#     save_path = '../pytorchStudy/test25/yolo_labels/all/labels/' + file_name_without_extension + '.txt'
#     file = open(save_path, 'w')

#     if clss[0] == 'Truck' :
#         file.write('0'+' ')
#     elif clss[0] == 'Bus' :
#         file.write('1'+' ')

#     # 计算中心点：
#     center_x = ( bbs[0][0] + bbs[0][2] ) / (2*width)
#     center_x = round(center_x, 2)
#     center_y = ( bbs[0][1] + bbs[0][3] ) / (2*height)
#     center_y = round(center_y, 2)
#     file.write(str(center_x)+' ')
#     file.write(str(center_y)+' ')

#     # 计算宽度和高度：
#     bbs_weight = ( bbs[0][2] - bbs[0][0] ) / width
#     bbs_weight = round(bbs_weight, 2)
#     bbs_height = ( bbs[0][3] - bbs[0][1] ) / height
#     bbs_height = round(bbs_height, 2)
#     file.write(str(bbs_weight)+' '+str(bbs_height)+'\n')
# file.close()
# print('数据保存完毕')


# print(clss)
def show_bbs(im, bbs, clss):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)
    for ix, (xmin, ymin, xmax, ymax) in enumerate(bbs):
        rect = mpatches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, 
                fill=False, 
                edgecolor='red', 
                linewidth=1)
        ax.add_patch(rect)
        centerx = xmin # + new_w/2
        centery = ymin + 20# + new_h - 10
        plt.text(centerx, centery, clss[ix],fontsize = 20,color='red')
    plt.show()

for i in range(20):
    im, bbs, clss, _ = ds[i]
    show_bbs(im, bbs, clss )


# (4)定义 extract_iou 和 extract_candidate 函数：
def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates

def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

"""
我们已经定义了准备数据和初始化数据加载器所需的所有函数，
在下一小节中，我们将获取区域提议(神经网络模型的输入区域)、真实的边界框偏移量以及对象类别(预期输出)。
"""


# 2.获取区域提议和偏移量
'''
在本节中，我们将学习如何创建模型相对应的输入和输出值。
模型输入使用选择性搜索提取的候选区域，输出包括候选区域的类别以及(如果候选区域包含目标对象)边界框的偏移量。
'''

# (1)初始化空列表以及存储文件路径(FPATHS)、真实边界框(GTBBS)、对象类别(CLSS)、边界框与区域提议之间的偏移量(DELTAS)、区域提议位置(ROIS)与真实边界框的交并比(IOUS)：
FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []


# (2)遍历数据集并填充初始化后的列表。
# 使用所有数据样本进行训练，也可以使用部分数据样本训练，数据样本越大，训练时间和准确度就越高：
N = 2000
for ix, (im, bbs, labels, fpath) in enumerate(ds):
    if(ix==N):
        break

    # 使用 extract_candidate() 函数从每个图像(im) 中提取候选区域，以绝对像素值表示(XMin、XMax、YMin、YMax 可作为图像形状的比例给出)，并将提取的区域坐标从 (x,y,w,h) 转换为 (x,y,x+w,y+h) 表示：
    H, W, _ = im.shape
    candidates = extract_candidates(im)
    candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])

    # 初始化 ious、rois、deltas 和 clss 为空列表，用于存储每个图像中每个候选区域与真实边界框的交并比、区域提议位置、边界框偏移量和每个候选区域的类别。
    # 遍历 SelectiveSearch 中的所有区域提议，并将那些具有较高 IoU 值且属于标签为 bus/truck 类别的区域提议存储为 bus/truck 提议，其余区域提议存储为背景提议：
    ious, rois, clss, deltas = [], [], [], []

    # 将所有候选区域与图像中所有真实边界框的交并比存储在 ious 中，其中 bbs 是图像中不同目标对象的真实边界框，candidates 是区域提议候选项：
    ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T

    # 遍历每个候选项并存储候选 XMin(cx)、YMin(cy)、XMax(xX) 和 YMax(cY) 值：
    for jx, candidate in enumerate(candidates):
        cx,cy,cX,cY = candidate

        # 提取与候选框相应的所有真实边界框的 IoU 值：
        candidate_ious = ious[jx]

        # 获取具有最高 IoU 的候选区域的索引(best_iou_at) 以及相应的真实边界框(best_bb)：
        best_iou_at = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_iou_at]
        best_bb = _x,_y,_X,_Y = bbs[best_iou_at]

        # 如果 IoU(best_iou) 大于给定阈值(0.3)，则为候选区域分配对应的类别标签，否则将其标记为背景：
        if best_iou > 0.3:
            clss.append(labels[best_iou_at])
        else:
            clss.append('background')

        # 获取所需的偏移量(delta) 以将当前区域提议转换为最佳区域提议对应的候选项 best_bb(即真实边界框)，
        # 换句话说，应调整当前提议坐标，才能使其完全与真实边界框 best_bb 对齐：
        delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
        deltas.append(delta)
        rois.append(candidate / np.array([W,H,W,H]))

    # 将文件路径、IoU、roi、类别、偏移量和真实边界框添加到结果列表中：
    # 注（前面也有）：存储文件路径(FPATHS)、真实边界框(GTBBS)、对象类别(CLSS)、边界框与区域提议之间的偏移量(DELTAS)、区域提议位置(ROIS)与真实边界框的交并比(IOUS)
    FPATHS.append(fpath)
    IOUS.append(ious)
    ROIS.append(rois)
    CLSS.append(clss)
    DELTAS.append(deltas)
    GTBBS.append(bbs)

# 获取图像路径名称并将获取到的所有信息———— FPATHS、IOUS、ROIS、CLSS、DELTAS 和 GTBBS 存储在列表中：
FPATHS, GTBBS, CLSS, DELTAS, ROIS = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]
'''
到目前为止，类别形式依旧是它们的名称，在神经网络模型训练过程中，需要将类别转换为对应的索引，
背景类别的索引为 0, 公交车类别的索引为 1, 卡车类别的索引为 2。
'''


# (3)为每个类别分配索引：
targets = pd.DataFrame([clss for l in CLSS for clss in l], columns=['label'])
label2target = {l:t for t,l in enumerate(targets['label'].unique())}
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']

print('数据准备完毕')

"""
我们已经为每个区域提议分配了一个类别，并创建了边界框偏移作为另一目标输出。
在下一节中，我们将获取与获得的信息(FPATHS、IOUS、ROIS、CLSS、DELTAS 和 GTBBS) 相对应的数据集和数据加载器。
"""


# 3.创建训练数据
'''
我们已经获取了所有的图像数据、区域提议，并得到了每个区域提议中目标对象的类别，以及区域提议与真实边界框相对应的偏移量，
这些区域提议与真实边界框具有较高的交并比 (Intersection over union, IoU)。
在本节中，我们将根据区域提议的真实标签准备数据集类，并从中创建数据加载器。

接下来，我们将每个区域提议调整为相同的形状并进行归一化。
'''

# (1)定义对图像执行归一化的函数：
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# (2)定义函数 preprocess_image() 预处理图像(img)，在该函数中，调整通道顺序，对图像进行归一化，并将其注册到设备中：
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()


# (3)定义函数解码类别预测结果：
def decode(_y):
    _, preds = _y.max(-1)
    return preds


# (4)使用预处理后的区域提议和真实标签，定义数据集类 RCNNDastaset：
class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
    def __len__(self):
        return len(self.fpaths)

    # 根据区域提议获取缩放图像，并获取与类别和边界框偏移相关的真实标签：
    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[...,::-1]
        H, W, _ = image.shape
        sh = np.array([W,H,W,H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois)*sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y,x:X] for (x,y,X,Y) in bbs]
        return image, crops, bbs, labels, deltas, gtbbs, fpath

    # 定义 collate_fn，执行对裁剪图像的缩放和归一化(preprocess_image)：
    def collate_fn(self, batch):
        input, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            crops = [preprocess_image(crop/255.)[None] for crop in crops]
            input.extend(crops)
            labels.extend([label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, labels, deltas


# (5)创建训练、验证数据集和数据加载器：
n_train = 9*len(FPATHS)//10
train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])
print(len(test_ds))
from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)

# 自我补充(为了yolo4训练，保存图像文件名称到train.txt 和 val.txt）
save_path = '../pytorchStudy/test25/yolo_labels/all/train.txt'
file = open(save_path, 'w')
for i in range(len(train_ds)):
    im, bbs, clss, _ = ds[i]
    full_name = os.path.basename(_)
    save_name = '../darknet/data/obj/images/'+full_name
    # file.write(full_name + '\n')
    file.write(save_name + '\n')
file.close()
print('数据保存完毕')

save_path = '../pytorchStudy/test25/yolo_labels/all/val.txt'
file = open(save_path, 'w')
for i in range(len(test_ds)):
    im, bbs, clss, _ = ds[i]
    full_name = os.path.basename(_)
    save_name = '../darknet/data/obj/images/'+full_name
    # file.write(full_name + '\n')
    file.write(save_name + '\n')
file.close()
print('数据保存完毕')


# 2.4构建 R-CNN 架构
'''
我们已经了解了如何准备数据，在本节中，我们将学习如何构建 R-CNN 目标检测模型用于预测区域提议、类别及其对应的偏移量，
以便在图像中的目标对象周围绘制边界框：
    1.定义 VGG 主干网络用于提取图像特征。
    2.使用预训练模型获取经过归一化缩放后的区域提议特征。
    3.在 VGG 主干网络上添加带有 sigmoid 激活的全连接层，以预测对应于区域提议的类别。
    4.添加另一全连接层预测边界框的四个偏移量。
    5.为以上两个输出(一个预测类别，另一个预测边界框的四个偏移量)定义损失函数。
    6.训练模型，预测区域提议的类别和边界框的四个偏移量。
'''

# (1)定义 VGG 主干网络：
vgg_backbone = models.vgg16(pretrained=True)
vgg_backbone.classifier = nn.Sequential()
for param in vgg_backbone.parameters():
    param.requires_grad = False
vgg_backbone.eval().to(device)

# (2)定义 R-CNN 网络模块。
# 定义模块类：
class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义主干网络(self.backbone)，以及输入分支计算类别分数(self.cls_score) 和边界框偏移值(self.bbox)：
        feature_dim = 25088
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )

        # 定义对应于类别预测(self.cel) 和边界框偏移回归(self.sl1) 的损失函数：
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    # 定义前向传播方法 forward，利用 VGG 主干网络(self.backbone) 获取图像特征(feat)，
    # 然后进一步将其通过分类和边界框回归方法传递，以获取类别概率(cls_score) 和边界框偏移量(bbox)：
    def forward(self, input):
        feat = self.backbone(input)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox

    # 定义损失函数(calc_loss)，如果真实类别为背景，不会计算与偏移量对应的回归损失：
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss


# (3)定义函数 train_batch()，用于在批数据上训练模型：
def train_batch(inputs, model, optimizer, criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()


# (4)定义函数 validate_batch()，用于验证模型：
@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss,_deltas = model(input)
        loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
        _, _clss = _clss.max(-1)
        accs = clss == _clss
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()


# (5)创建模型对象，获取损失，然后定义优化器和训练 epoch 数：
rcnn = RCNN().to(device)
criterion = rcnn.calc_loss
optimizer = optim.SGD(rcnn.parameters(), lr=1e-3)
n_epochs = 10


# (6)训练模型：
train_loss_epochs = []
train_loc_loss_epochs = []
train_regr_loss_epochs = []
train_acc_epochs = []
val_loc_loss_epochs = []
val_regr_loss_epochs = []
val_loss_epochs = []
val_acc_epochs = []
for epoch in range(n_epochs):
    train_loss = []
    train_loc_loss = []
    train_regr_loss = []
    train_acc = []
    val_loc_loss = []
    val_regr_loss = []
    val_loss = []
    val_acc = []
    _n = len(train_loader)
    start_time = time.time()

    for ix, inputs in enumerate(train_loader):
        loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, 
                                                      optimizer, criterion)
        pos = (epoch + (ix+1)/_n)
        train_loss.append(loss.item())
        train_loc_loss.append(loc_loss.item())
        train_regr_loss.append(regr_loss.item())
        train_acc.append(accs.mean())
    train_loss_epochs.append(np.average(train_loss))
    train_loc_loss_epochs.append(np.average(train_loc_loss))
    train_regr_loss_epochs.append(np.average(train_regr_loss))
    train_acc_epochs.append(np.average(train_acc))
        
    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        _clss, _deltas, loss, \
        loc_loss, regr_loss, accs = validate_batch(inputs, 
                                                rcnn, criterion)
        pos = (epoch + (ix+1)/_n)
        val_loss.append(loss.item())
        val_loc_loss.append(loc_loss.item())
        val_regr_loss.append(regr_loss.item())
        val_acc.append(accs.mean())
    val_loss_epochs.append(np.average(val_loss))
    val_loc_loss_epochs.append(np.average(val_loc_loss))
    val_regr_loss_epochs.append(np.average(val_regr_loss))
    val_acc_epochs.append(np.average(val_acc))

    end_time = time.time()
    print(f"epoch:{epoch+1}/{n_epochs}, time:{end_time-start_time}")


# (7)可视化模型训练过程中训练和验证数据集中的损失值和准确率的变化：
epochs = np.arange(10)+1
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.subplot(221)
plt.plot(epochs, train_loss_epochs, 'bo', label='Training loss')
plt.plot(epochs, val_loss_epochs, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')

plt.subplot(222)
plt.plot(epochs, train_loc_loss_epochs, 'bo', label='Training Loss')
plt.plot(epochs, val_loc_loss_epochs, 'r', label='Validation Loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')

plt.subplot(223)
plt.plot(epochs, train_regr_loss_epochs, 'bo', label='Training loss')
plt.plot(epochs, val_regr_loss_epochs, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')

plt.subplot(224)
plt.plot(epochs, train_acc_epochs, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_epochs, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss when batch size is 32')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().yaxis.set_major_locator(mticker.FixedLocator(plt.gca().get_yticks()))
ticks_loc = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in ticks_loc])
plt.legend()
plt.grid('off')

plt.show()


# (8)模型保存
torch.save(rcnn.state_dict(), "../pytorchStudy/test25/myrcnn_model.pth")