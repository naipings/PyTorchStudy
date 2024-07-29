# 2D 和 3D 面部关键点检测

'''
在上一小节中，我们从零开始构建了面部关键点检测器模型。
在本节中，我们将学习如何利用专门为 2D 和 3D 关键点检测而构建的预训练模型来获取面部的 2D 和 3D 关键点。
为了完成此任务，我们将使用 face-alignment 库。
'''

# (1)使用 pip 安装 face-alignment 库：
# pip install face-alignment

# (2)加载所需的库：
import face_alignment
import cv2

file = '../pytorchStudy/test22/22-2imgs/test2.jpg'
im = cv2.imread(file)
input = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# (3)定义人脸对齐方法，指定是要获取 2D 还是 3D 关键点坐标：
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

# (4)读取输入图像并将其作为 get_landmarks 方法的输入：
preds = fa.get_landmarks(input)[0]
print(preds.shape)
# (68, 2)
'''
在以上代码中，利用 FaceAlignment 类的对象 fa 中的 get_landmarks 方法来获取与面部关键点对应的 68 个 x 和 y 坐标。
'''

# (5)用检测到的关键点绘制图像：
import matplotlib
matplotlib.use('TkAgg') # 在一个新窗口打开图形
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5))
plt.imshow(input)
ax.scatter(preds[:,0], preds[:,1], marker='+', c='r')
plt.show()

# (6)获取面部关键点的 3D 投影：
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
im = cv2.imread(file)
input = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
preds = fa.get_landmarks(input)[0]
import pandas as pd
df = pd.DataFrame(preds)
df.columns = ['x','y','z']
import plotly.express as px
fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z')
fig.show()
'''
与 2D 关键点检测中使用的代码的唯一变化是将 LandmarksType 指定为 3D 而不是 2D, 输出结果见效果图。
通过利用 face_alignment 库，可以看到利用预训练的面部关键点检测模型在预测新图像时具有较高的精度。
'''