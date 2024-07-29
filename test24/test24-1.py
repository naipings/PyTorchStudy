# 目标检测基础
# ————3.区域提议


# ————3.2 利用 SelectiveSearch 生成区域提议
'''
选择性搜索 (Selective Search) 是一种经典的区域提议算法，用于生成可能包含目标物体的候选区域，
基于图像分割和区域合并的思想，通过逐步合并相似的区域来生成候选区域。
它的基本思想是通过对图像进行分层分组，生成不同尺度和大小的图像区域。
这些区域被视为候选检测区域，以便后续的检测器可以针对这些区域进行进一步处理。
Selective Search 算法通过以下操作生成候选区域：
    1.将图像分割成多个区域，每个区域由相似的颜色、纹理和结构特征组成
    2.使用图像分割结果形成初始的候选区域集合
    3.计算候选区域之间的相似度，并将相似度最高的候选区域合并成更大的区域，更新候选区域集合
    4.SelectiveSearch 根据不同尺度和比例下的相似性将区域进行多次合并，得到一组更精细的候选区域
    5.重复执行第 3 和第 4 步，得到一组不重叠且可靠的候选区域

SelectiveSearch 算法的优点在于它可以生成高质量的候选区域，并且具有较好的鲁棒性，适用于多种目标检测任务。
接下来，我们使用 Python 实现选择性搜索的过程。

在 Python 中, Selective Search for Object Recognition (selectivesearch) 是常用的选择性搜索算法库，
能够方便地使用选择性搜索算法生成候选区域。
'''

# (1)安装所需的库：
# pip install selectivesearch

# (2)加载图像以及所需的库：
import selectivesearch
from skimage.segmentation import felzenszwalb
import cv2
import matplotlib
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
import numpy as np

img_r = cv2.imread('../pytorchStudy/test24/24-1imgs/test2.jpg')
img = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# (3)基于图像的颜色、纹理、大小和形状，从图像中提取 felzenszwalb 分割：
segments_fz = felzenszwalb(img, scale=200)
'''
在 felzenszwalb 方法中, scale 代表可以在图像分割中形成的聚类数, scale 的数值越高，保留原始图像细节的程度就越高。
换句话说, scale 值越高，生成的分割内容越精细。
'''

# (4)绘制原始图像和分割后的图像：
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(122)
plt.imshow(segments_fz)
plt.title('Image post \nfelzenszwalb segmentation')
plt.show()
'''
根据结果图可知，属于同一组的像素在分割结果图中具有相似的像素值。具有相似值的像素构成一个区域提议。
使用区域提议有助于目标检测，因为我们可以将每个区域提议传递给网络并预测区域提议是背景还是目标对象。
此外，如果区域提议是一个目标对象，该区域可以用于识别偏移量以获取与对象边界框以及与区域提议中的内容对应的类别。
了解了 SelectiveSearch 算法原理后，我们使用选择性搜索函数来获取给定图像的区域提议。
'''



# ————3.3 生成区域提议
''' 在本节中，我们将使用选择性搜索定义 extract_candidates 函数，以便为后续的目标检测模型训练奠定基础。 '''

# (1)定义从图像中提取区域提议的函数 extract_candidates()：
# 将图像作为输入参数：
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

# (2)导入相关库与图像：
img = cv2.imread('../pytorchStudy/test24/24-1imgs/test2.jpg')
candidates = extract_candidates(img)

# (3)提取候选区域在图像上可视化：
import matplotlib.patches as mpatches
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
for x, y, w, h in candidates:
    rect = mpatches.Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor='blue',
            linewidth=4)
    ax.add_patch(rect)
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
'''
通过结果图可知，图中的网格表示使用 selective_search 方法获取的区域提议(候选区域)。
'''

"""
我们已经了解了如何生成区域提议，接下来，继续学习如何利用区域提议进行目标检测和定位。
一个区域提议如果与图像中的任意一个目标对象位置有很高的重合面积，就会被标记为包含该对象的提议，
而与其交集很小的区域提议将被标记为背景。在下一节中，我们将介绍如何计算候选区域与真实边界框的交集。
"""