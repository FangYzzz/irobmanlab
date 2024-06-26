import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 读取图像
# image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("/home/yuan/Mani-GPT/saucepanhandle.png")


def edge(image, cx, cy, w, h):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cx = int(cx)
    cy = int(cy)
    w = int(w)
    h = int(h)

    # 定义感兴趣区域 (ROI)，这里假设ROI是矩形区域，使用顶点坐标 (x, y, w, h) 定义
    roicopy = gray_image[cy:cy+h, cx:cx+w]
    # roicopy = gray_image
    roicopy = np.uint8(roicopy)

    # 应用边缘检测（Canny边缘检测算法）
    edges = cv2.Canny(roicopy, 100, 200)
    print("Data type of edges:", edges.dtype)

    # 创建一个与原图大小相同的全黑图像
    edge_image = np.zeros_like(gray_image)
    # 将ROI边缘结果放回到相应位置
    edge_image[cy:cy+h, cx:cx+w] = edges
    # edge_image = edges

    # 显示结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original GrayImage')
    plt.imshow(gray_image, cmap='gray')

    # edgescopy = np.uint8(edges)
    # plt.subplot(1, 3, 2)
    # plt.title('Edge Detection on ROI')
    # plt.imshow(edges, cmap='gray')

    edge_imagecopy = np.uint8(edge_image)
    plt.subplot(1, 3, 3)
    plt.title('Edges on Original Image')
    plt.imshow(edge_image, cmap='gray')


# edge(image, 100,100,50,50)
# plt.show()




# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # 读取图像
# image = cv2.imread('/home/yuan/Mani-GPT/saucepanhandle.png', 0)  # 读取图像并转换为灰度图

# # # 使用Sobel算子
# # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X方向梯度
# # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y方向梯度

# # # 计算梯度的绝对值
# # abs_sobel_x = cv2.convertScaleAbs(sobel_x)
# # abs_sobel_y = cv2.convertScaleAbs(sobel_y)

# # # 合并梯度
# # sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# # 使用Canny算子
# edges = cv2.Canny(image, 120, 240)

# # 使用Matplotlib显示结果
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# # plt.subplot(2, 2, 2), plt.imshow(sobel_combined, cmap='gray')
# # plt.title('Sobel Edge Detection'), plt.xticks([]), plt.yticks([])

# plt.subplot(2, 2, 3), plt.imshow(edges, cmap='gray')
# plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

# plt.show()
