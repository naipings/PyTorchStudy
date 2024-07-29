# R-CNN 目标检测模型测试
'''
在本节中，我们将利用训练后的 R-CNN 模型来预测和绘制目标对象边界框以及边界框内的目标对象类别：
    1.在测试图像中提取区域提议。
    2.调整每个区域提议的大小并进行归一化。
    3.将经过处理的区域提议图像通过前向传播后预测类别和偏移量。
    4.执行非极大值抑制，仅获取具有包含对象的具有最高置信度的边界框。
'''

"""
注：数据集准备、获取区域提议和偏移量、创建训练数据、构建 R-CNN 架构：
均为test25-1.py 中的内容并做了少量修改(移除了训练步骤，直接加载上一节已经保存的训练好的模型)。
"""

# 1.数据集准备
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
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_ROOT = '../pytorchStudy/test25/open-images-bus-trucks/images/images'
DF_RAW = pd.read_csv('../pytorchStudy/test25/open-images-bus-trucks/df.csv')
print(DF_RAW.head())


# (2)定义类 OpenImages，返回图像及其包含的目标对象的类别、目标对象边界框以及图像的文件路径。
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

''' 本节内容：(1)修改 show_bbs() 函数用于可视化 R-CNN 检测结果：'''
def show_bbs(im, bbs, clss, ax):
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)
    for ix, (xmin, ymin, xmax, ymax) in enumerate(bbs):
        rect = mpatches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, 
                fill=False, 
                edgecolor='red', 
                linewidth=1)
        ax.add_patch(rect)
        centerx = xmin # + new_w/2
        centery = ymin + 20 # + new_h - 10
        plt.text(centerx, centery, clss[ix],fontsize = 20,color='red')


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


# 2.获取区域提议和偏移量
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

# 获取图像路径x名称并将获取到的所有信息———— FPATHS、IOUS、ROIS、CLSS、DELTAS 和 GTBBS 存储在列表中：
FPATHS, GTBBS, CLSS, DELTAS, ROIS = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]


# (3)为每个类别分配索引：
targets = pd.DataFrame([clss for l in CLSS for clss in l], columns=['label'])
label2target = {l:t for t,l in enumerate(targets['label'].unique())}
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']

print('数据准备完毕')


# 3.创建训练数据
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
test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])
print(len(test_ds))
from torch.utils.data import TensorDataset, DataLoader
test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)


# 4.构建 R-CNN 架构
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


# (3)加载上一节已经训练好的模型
rcnn = RCNN().to(device)
rcnn.load_state_dict(torch.load("../pytorchStudy/test25/myrcnn_model.pth"))


''' 本节内容：(2)定义函数 test_predictions() 在测试图像上进行预测。'''
# 函数将文件名作为输入：
def test_predictions(filename, show_output=True):
    # 读取图像并提取候选区域：
    img = np.array(cv2.imread(filename, 1)[...,::-1])
    candidates = extract_candidates(img)
    candidates = [(x,y,x+w,y+h) for x,y,w,h in candidates]

    # 循环遍历候选区域，调整图像大小并预处理图像：
    input = []
    for candidate in candidates:
        x,y,X,Y = candidate
        crop = cv2.resize(img[y:Y,x:X], (224,224))
        input.append(preprocess_image(crop/255.)[None])
    input = torch.cat(input).to(device)

    # 预测类别和偏移量：
    with torch.no_grad():
        rcnn.eval()
        probs, deltas = rcnn(input)
        probs = torch.nn.functional.softmax(probs, -1)
        confs, clss = torch.max(probs, -1)

    # 提取不属于背景类别的候选区域，并将候选区域与预测的边界框偏移值相加得到预测边界框：
    candidates = np.array(candidates)
    confs, clss, probs, deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, probs, deltas]]

    ixs = clss!=background_class
    confs, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates]]
    bbs = (candidates + deltas).astype(np.uint16)

    # 使用非极大值抑制(non-maxium suppression, NMS) 消除重复边界框(IoU 大于 0.05 的边界框可以认为是重复的)，在重复的边界框中，选择置信度最高的边界框，并丢弃其余边界框：
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    confs, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates, bbs]]
    if len(ixs) == 1:
        confs, clss, probs, deltas, candidates, bbs = [tensor[None] for tensor in [confs, clss, probs, deltas, candidates, bbs]]

    # 获取置信度最高的边界框：
    if len(confs) == 0 and not show_output:
        return (0,0,224,224), 'background', 0
    if len(confs) > 0:
        best_pred = np.argmax(confs)
        best_conf = np.max(confs)
        best_bb = bbs[best_pred]
        x,y,X,Y = best_bb

    # 绘制图像与预测边界框：
    _, ax = plt.subplots(1, 2, figsize=(20,10))
    ax[0].imshow(img)
    # show(img, ax=ax[0])
    ax[0].grid(False)
    ax[0].set_title('Original image')
    print(len(confs))
    if len(confs) == 0:
        ax[1].imshow(img)
        ax[1].set_title('No objects')
        plt.show()
        return
    ax[1].set_title(target2label[clss[best_pred]])
    show_bbs(img, bbs=bbs.tolist(), clss=[target2label[c] for c in clss.tolist()], ax=ax[1])
    # ax[1].title('predicted bounding box and class')
    plt.show()
    return (x,y,X,Y),target2label[clss[best_pred]],best_conf


''' 本节内容：(3)在测试图像上执行函数 test_predictions() : '''
for i in range(30):
    random_num = random.randint(0, 199)
    image, crops, bbs, labels, deltas, gtbbs, fpath = test_ds[random_num]
    test_predictions(fpath)


"""
使用测试图像生成预测结果大约需要 1.5 秒，大部分时间用于生成区域提议、调整每个区域提议的尺寸、将它们输入到 VGG 主干网络、使用训练后的模型生成预测结果。

R-CNN 是基于候选区域的经典目标检测算法，其将卷积神经网络引入目标检测领域，其思想和方法为后续的目标检测算法发展奠定了基础。
尽管 R-CNN 在目标检测领域取得了很大的成功，但因为它需要逐个处理候选区域，导致其速度较慢。
本节(test25) 首先介绍了 R-CNN 模型的核心思想与目标检测流程，然后使用 PyTorch 从零开始实现了一个基于 R-CNN 的目标检测模型。
"""