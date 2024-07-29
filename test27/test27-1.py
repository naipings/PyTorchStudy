# 实现 Faster R-CNN 目标检测

'''

'''

# 1.数据处理

# (1)读取包含图像及其边界框和类别信息元数据的 DataFrame ：
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
from PIL import Image
import time
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_ROOT = '../pytorchStudy/test25/open-images-bus-trucks/images/images'
DF_RAW = pd.read_csv('../pytorchStudy/test25/open-images-bus-trucks/df.csv')
print(DF_RAW.head())


# (2)定义标签对应的索引：
label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)


# (3)定义图像预处理函数 preprocess_image() 与图像查找函数 find() ：
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()

def find(item, original_list):
    results = []
    for o_i in original_list:
        if item in o_i:
            results.append(o_i)
    if len(results) == 1:
        return results[0]
    else:
        return results


# (4)定义数据集类 OpenDataset。
# 定义 __init__ 方法，将图像文件夹和包含图像元数据的 DataFrame 作为输入：
class OpenDataset(torch.utils.data.Dataset):
    w, h = 224, 224
    def __init__(self, df, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob(self.image_dir+'/*')
        self.df = df
        self.image_infos = df.ImageID.unique()

    # 定义 __getitem__ 方法，返回预处理后的图像和目标值：
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        df = self.df.copy()
        data = df[df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
        # torch FRCNN expects ground truths as a dictionary of tensors
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([label2target[i] for i in labels]).long()
        img = preprocess_image(img)
        return img, target
    ''' 以上 __getitem__ 方法将输出作为张量字典而非张量列表返回，这是因为我们期望输出包含边界框的绝对坐标和标签信息。 '''

    # 定义 collate_fn 方法(处理字典列表) 和 __len__ 方法：
    def collate_fn(self, batch):
        return tuple(zip(*batch)) 

    def __len__(self):
        return len(self.image_infos)


# (5)创建训练和验证数据集和数据加载器：
from sklearn.model_selection import train_test_split
df = DF_RAW.copy()
trn_ids, val_ids = train_test_split(df.ImageID.unique(), test_size=0.1, random_state=99)
trn_df, val_df = df[df['ImageID'].isin(trn_ids)], df[df['ImageID'].isin(val_ids)]
len(trn_df), len(val_df)
# print(len(trn_df), len(val_df))

train_ds = OpenDataset(trn_df)
test_ds = OpenDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)


# 2.模型构建

# (1)定义模型：
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
'''

'''


# (2)定义函数在批数据上训练网络，并计算验证数据集的损失值：
def train_batch(inputs, model, optimizer):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses

@torch.no_grad()
def validate_batch(inputs, model, optimizer):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses


# 3.模型训练与测试

# (1)训练模型。
# 初始化模型：
model = get_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
n_epochs = 10

train_loss_epochs = []
train_loc_loss_epochs = []
train_regr_loss_epochs = []
train_objectness_loss_epochs = []
train_rpn_box_reg_loss_epochs = []
val_loss_epochs = []
val_loc_loss_epochs = []
val_regr_loss_epochs = []
val_objectness_loss_epochs = []
val_rpn_box_reg_loss_epochs = []

# 训练模型并计算训练和测试数据集的损失值：
for epoch in range(n_epochs):
    _n = len(train_loader)
    trn_loss = []
    trn_loc_loss = []
    trn_regr_loss = []
    trn_objectness_loss = []
    trn_rpn_box_reg_loss = []
    val_loss = []
    val_loc_loss = []
    val_regr_loss = []
    val_objectness_loss = []
    val_rpn_box_reg_loss = []
    start_time = time.time()

    for ix, inputs in enumerate(train_loader):
        loss, losses = train_batch(inputs, model, optimizer)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
            [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
        pos = (epoch + (ix+1)/_n)
        trn_loss.append(loss.item())
        trn_loc_loss.append(loc_loss.item())
        trn_regr_loss.append(regr_loss.item())
        trn_objectness_loss.append(loss_objectness.item())
        trn_rpn_box_reg_loss.append(loss_rpn_box_reg.item())
    train_loss_epochs.append(np.average(trn_loss))
    train_loc_loss_epochs.append(np.average(trn_loc_loss))
    train_regr_loss_epochs.append(np.average(trn_regr_loss))
    train_objectness_loss_epochs.append(np.average(trn_objectness_loss))
    train_rpn_box_reg_loss_epochs.append(np.average(trn_rpn_box_reg_loss))

    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        loss, losses = validate_batch(inputs, model, optimizer)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
          [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
        pos = (epoch + (ix+1)/_n)
        val_loss.append(loss.item())
        val_loc_loss.append(loc_loss.item())
        val_regr_loss.append(regr_loss.item())
        val_objectness_loss.append(loss_objectness.item())
        val_rpn_box_reg_loss.append(loss_rpn_box_reg.item())
    val_loss_epochs.append(np.average(val_loss))
    val_loc_loss_epochs.append(np.average(val_loc_loss))
    val_regr_loss_epochs.append(np.average(val_regr_loss))
    val_objectness_loss_epochs.append(np.average(val_objectness_loss))
    val_rpn_box_reg_loss_epochs.append(np.average(val_rpn_box_reg_loss))

    end_time = time.time()
    print(f"epoch:{epoch+1}/{n_epochs}, time:{end_time-start_time}")


# (2)绘制损失值属于训练的变化情况：
epochs = np.arange(n_epochs)+1
plt.plot(epochs, train_loss_epochs, 'bo', label='Training loss')
plt.plot(epochs, val_loss_epochs, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

# 模型保存：
torch.save(model.state_dict(), "../pytorchStudy/test27/myferrcnn_model.pth")

# (3)使用训练后的 Faster R-CNN 模型预测测试图像。
'''

'''
from torchvision.ops import nms
def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

# print(clss)
def show_bbs(im, bbs, clss):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 6))
    ax[0].imshow(im)
    ax[0].grid(False)
    ax[0].set_title('Original image')
    if len(bbs) == 0:
        ax[1].imshow(im)
        ax[1].set_title('No objects')
        plt.show()
        return
    ax[1].imshow(im)
    for ix, (xmin, ymin, xmax, ymax) in enumerate(bbs):
        rect = mpatches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, 
                fill=False, 
                edgecolor='red', 
                linewidth=1)
        ax[1].add_patch(rect)
        centerx = xmin # + new_w/2
        centery = ymin + 20# + new_h - 10
        plt.text(centerx, centery, clss[ix],fontsize = 20,color='red')
    ax[1].grid(False)
    ax[1].set_title('Predicted bounding box and class')
    plt.show()


# 获取测试图像中目标对象的边界框和类别：
model.eval()
for ix, (images, targets) in enumerate(test_loader):
    if ix==20: 
        break
    images = [im for im in images]
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
        info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
        show_bbs(images[ix].cpu().permute(1,2,0), bbs=bbs, clss=labels)

