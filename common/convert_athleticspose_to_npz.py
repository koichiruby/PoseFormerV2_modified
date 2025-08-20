import os
import numpy as np

# 配置路径
dataset_root = r"D:\pycharmprojects\PoseFormerV2\AthleticsPoseDataset"
pose2d_root = os.path.join(dataset_root, "det_markers2d_by_cam_coco")  # 你也可以改成 det_markers2d_by_cam_ft
pose3d_root = os.path.join(dataset_root, "gt_markers3d_by_cam")

output_dir = os.path.join(dataset_root, "converted_npz")
os.makedirs(output_dir, exist_ok=True)

def load_3d_file(base_path_without_ext):
    npy_path = base_path_without_ext + ".npy"
    npz_path = base_path_without_ext + ".npz"
    if os.path.isfile(npy_path):
        return np.load(npy_path)
    elif os.path.isfile(npz_path):
        try:
            with np.load(npz_path) as data:
                key = data.files[0]  # 默认第一个 key
                return data[key]
        except Exception as e:
            print(f"[警告] 加载 npz 文件失败，尝试作为 npy 加载: {npz_path}，错误: {e}")
            # 尝试当作 npy 文件加载
            try:
                return np.load(npz_path, allow_pickle=True)
            except Exception as e2:
                print(f"[错误] 作为 npy 加载也失败: {npz_path}，错误: {e2}")
                return None
    else:
        return None


def convert_and_save(sport, subject, filename):
    file_2d = os.path.join(pose2d_root, sport, subject, filename)
    base_3d = os.path.join(pose3d_root, sport, subject, filename[:-4])  # 去掉 .npy 后缀

    if not os.path.isfile(file_2d):
        print(f"[跳过] 不存在2D文件: {file_2d}")
        return

    kp2d = np.load(file_2d)

    kp3d = load_3d_file(base_3d)
    if kp3d is None:
        print(f"[警告] 不存在3D文件，使用零填充: {base_3d}(.npy/.npz)")
        kp3d = np.zeros((kp2d.shape[0], kp2d.shape[1], 3), dtype=np.float32)

    base_name = f"{sport}_{subject}_{filename.replace('.npy', '')}.npz"
    out_path = os.path.join(output_dir, base_name)

    np.savez_compressed(out_path, positions_2d=kp2d, positions_3d=kp3d)
    print(f"[完成] {out_path}")

def main():
    for sport in os.listdir(pose2d_root):
        sport_path = os.path.join(pose2d_root, sport)
        if not os.path.isdir(sport_path):
            continue

        for subject in os.listdir(sport_path):
            subject_path = os.path.join(sport_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for filename in os.listdir(subject_path):
                if not filename.endswith(".npy"):
                    continue

                convert_and_save(sport, subject, filename)

if __name__ == "__main__":
    main()
