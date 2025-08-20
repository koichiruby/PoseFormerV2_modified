import os
import numpy as np

# === 配置路径 ===
converted_dir = r"D:\pycharmprojects\PoseFormerV2\AthleticsPoseDataset\converted_npz"
output_2d_path = r"D:\pycharmprojects\PoseFormerV2\data\data_2d_athletics.npz"
output_3d_path = r"D:\pycharmprojects\PoseFormerV2\data\data_3d_athletics.npz"

def merge_npz_files():
    positions_2d_dict = {}
    positions_3d_dict = {}

    files = [f for f in os.listdir(converted_dir) if f.endswith(".npz")]
    print(f"[INFO] 找到 {len(files)} 个 npz 文件进行合并")

    for fname in files:
        # sport_S01_20250125_00_0.npz
        parts = fname.split("_", 2)
        if len(parts) < 3:
            print(f"[跳过] 文件名格式不符: {fname}")
            continue

        sport = parts[0]
        subject_id = parts[1]
        action = sport

        file_path = os.path.join(converted_dir, fname)
        with np.load(file_path) as data:
            kp2d = data["positions_2d"]  # (T, J, 2)
            kp3d = data["positions_3d"]  # (T, J, 3)

        if subject_id not in positions_2d_dict:
            positions_2d_dict[subject_id] = {}
            positions_3d_dict[subject_id] = {}

        if action not in positions_2d_dict[subject_id]:
            positions_2d_dict[subject_id][action] = []
            positions_3d_dict[subject_id][action] = []

        # 直接 append，不做 np.array
        positions_2d_dict[subject_id][action].append(kp2d)
        positions_3d_dict[subject_id][action].append(kp3d)

        print(f"[合并] {fname} → subject={subject_id}, action={action}, frames={kp2d.shape[0]}")

    # 保存成 object 数组（和官方 Human3.6M 一样）
    np.savez_compressed(output_2d_path, positions_2d=positions_2d_dict)
    np.savez_compressed(output_3d_path, positions_3d=positions_3d_dict)

    print(f"[完成] 已保存 2D 数据到 {output_2d_path}")
    print(f"[完成] 已保存 3D 数据到 {output_3d_path}")

if __name__ == "__main__":
    merge_npz_files()
