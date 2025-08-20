import numpy as np

# 读取 npy 文件
data = np.load(r"D:\pycharmprojects\PoseFormerV2\AthleticsPoseDataset\det_markers2d_by_cam_coco\discus\S00\20250125_00_0.npy")
print(type(data))  # 看看是 ndarray 还是 dict（如果是 npz）
print(data.shape)  # 输出 (T, J, C) 或其他形状
