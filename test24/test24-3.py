# 目标检测基础
# ————5.非极大值抑制

# 加载图像以及所需的库：
import selectivesearch
from skimage.segmentation import felzenszwalb
import cv2
import matplotlib
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

# 生成区域提议
def extract_candidates(img):
    # 使用 selectivesearch 库中提供的 selective_search 方法获取图像中的候选区域：
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=2000)
    # 计算图像区域并初始化一个列表(候选区域)，使用该列表来存储通过定义阈值的候选区域：
    candidates = []
    # 获取图像总面积：
    img_area = img.shape[0] * img.shape[1]
    print("Total area of the image is:", img_area)
    # 仅获取超过图像总面积 5% 且不超过图像面积 100% 的区域作为候选区域并返回：
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < (0.05*img_area):
            continue
        if r['size'] > (1*img_area):
            continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates

# 读取存储原始图像真实标记框坐标的 xml 文件
import xml.dom.minidom as xmldom
def get_bndboxfromxml():
# def get_bndboxfromxml(imageNum, xmlfilebasepath):
    # 读取xml文件
    bndbox = [0, 0, 0, 0]
    # xmlfilepath = xmlfilebasepath + "\%06d" % imageNum+'.xml'
    xmlfilepath = '../pytorchStudy/test24/24-3imgs/test2.xml'
    # print(xmlfilepath)
    domobj = xmldom.parse(xmlfilepath)
    elementobj = domobj.documentElement
    sub_element_obj = elementobj.getElementsByTagName('bndbox')
    if sub_element_obj is not None:
        bndbox[0] = int(sub_element_obj[0].getElementsByTagName('xmin')[0].firstChild.data)
        bndbox[1] = int(sub_element_obj[0].getElementsByTagName('ymin')[0].firstChild.data)
        bndbox[2] = int(sub_element_obj[0].getElementsByTagName('xmax')[0].firstChild.data)
        bndbox[3] = int(sub_element_obj[0].getElementsByTagName('ymax')[0].firstChild.data)
    return bndbox

def get_iou(boxA, boxB, epsilon=1e-5):
# 我们需要额外定义 epsilon 参数来解决两个边界框之间的并集为 0 时的情况。以避免出现除零错误。
    # 计算交集框坐标：
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    # 计算相交区域(重叠区域)对应的宽和高：
    width = (x2 - x1)
    height = (y2 - y1)
    # 计算重叠面积(area_overlap)：
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    # 计算两个边界框对应的组合面积：
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    # 计算 IoU 并返回：
    iou = area_overlap / (area_combined+epsilon)
    return iou

# 导入相关库与图像：
img = cv2.imread('../pytorchStudy/test24/24-1imgs/test2.jpg')
candidates = extract_candidates(img)
# print(candidates)
bndbox = get_bndboxfromxml()
bndbox_tensor = torch.tensor(bndbox)

# 提取候选区域在图像上可视化：
# 直接使用 iou 筛选合适的特征框（只能初筛）
import matplotlib.patches as mpatches
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
i = 0
for x, y, w, h in candidates:
    iou = get_iou(bndbox, candidates[i])
    # print(iou)
    if iou > 0.5:
        rect = mpatches.Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor='blue',
            linewidth=4)
        ax.add_patch(rect)
    i = i+1
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# 以下代码还有些问题：
# 使用 torchvision.ops 中的 nms 函数执行非极大值抑制
# import matplotlib.patches as mpatches
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# pred = 0.91
# pred_tensor = torch.tensor(pred)
# i = 0
# for x, y, w, h in candidates:
#     candidates_new = []
#     length = int(len(candidates[i])/2)
#     candidates_new=[ [ candidates[i][0],candidates[i][1] ], [ candidates[i][2],candidates[i][3] ] ]
#     # print(candidates_new)

#     iou = get_iou(bndbox, candidates[i])
#     # print(iou)
#     candidates_tenser = torch.tensor(candidates_new)
#     a = torchvision.ops.nms(candidates_tenser, pred_tensor, iou)
#     # if iou > 0.5:
#     #     rect = mpatches.Rectangle(
#     #         (x, y), w, h,
#     #         fill=False,
#     #         edgecolor='blue',
#     #         linewidth=4)
#     #     ax.add_patch(rect)
#     print(a)
#     i = i+1
# # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# # plt.show()