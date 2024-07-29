# 交并比
# ————4.实现 IoU 计算函数
''' 定义函数，将两个边界框作为输入并返回 IoU 作为输出。 '''

# (1)定义 get_iou() 函数，将 boxA 和 boxB 作为输入，其中 boxA 和 boxB 是两个不同的边界框(可以将 boxA 视为真实边界框，将 boxB 视为区域提议)：
def get_iou(boxA, boxB, epsilon=1e-5):
# 我们需要额外定义 epsilon 参数来解决两个边界框之间的并集为 0 时的情况。以避免出现除零错误。

    # (2)计算交集框坐标：
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    '''
    x1 存储了两个边界框之间最左侧 x 坐标的最大值，
    y1 存储了最上面的 y 坐标的最大值，
    x2 和 y2 分别存储了两个边界框之间最右边的 x 坐标和最底部的 y 坐标的最小值，
    对应于两个边界框的相交部分。
    '''

    # (3)计算相交区域(重叠区域)对应的宽和高：
    width = (x2 - x1)
    height = (y2 - y1)

    # (4)计算重叠面积(area_overlap)：
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    '''
    在以上代码中，指定如果重叠区域对应的宽度或高度小于 0, 则相交的面积为 0。
    否则，重叠(相交)面积等于相交区域的宽度乘以高度。
    '''

    # (5)计算两个边界框对应的组合面积：
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    '''
    在以上代码中，计算两个边界框的组合面积，首先计算 area_a+area_b, 
    然后计算 area_combined 时减去重叠区域 area_overlap, 
    因为 area_overlap 被计算两次(计算 area_a 时计算一次，计算 area_b 时计算一次)。
    '''

    # 计算 IoU 并返回：
    iou = area_overlap / (area_combined+epsilon)
    return iou
    '''
    在以上代码中，将 iou 计算为重叠面积(area_overlap) 与组合面积(area_combined) 之比并返回。
    '''