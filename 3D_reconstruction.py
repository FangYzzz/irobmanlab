import numpy as np
import cv2

# 相机标定参数
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
D = np.array([k1, k2, p1, p2])
R = np.array([[r1, r2, r3], [r4, r5, r6], [r7, r8, r9]])
T = np.array([tx, ty, tz])

# 读取左右两张图像
imgL = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化SGBM立体匹配器
window_size = 15
min_disp = 0
num_disp = 160
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=15,
                               speckleWindowSize=100,
                               speckleRange=32)

# 计算左右两张图像的视差图
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# 使用相机标定参数和视差图进行三维重建
points = cv2.stereoReconstruct(K, D, R, T, K, (imgL.shape[1], imgL.shape[0]), disp)


# 可视化结果
cv2.imshow('reconstruction', (points[1:, :] / points[1:, 2:3]).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()