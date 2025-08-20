import cv2
import numpy as np
from pathlib import Path
from mmpose.apis import init_pose_model, inference_top_down_pose_model

# 初始化HRNet模型（CPU版或CUDA版）
pose_config = 'configs/hrnet/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-b9e0b3ab_20200708.pth'
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cpu')

def simple_fullbody_bbox(frame):
    h, w, _ = frame.shape
    return np.array([0, 0, w - 1, h - 1])

def extract_keypoints(frame, bbox):
    person = {'bbox': bbox}
    pose_results, _ = inference_top_down_pose_model(pose_model, frame, [person], bbox_thr=None, format='xyxy')
    if len(pose_results) > 0:
        return pose_results[0]['keypoints'][:, :2]
    else:
        return np.zeros((17, 2), dtype=np.float32)

video_dir = Path(r"D:\pycharmprojects\PoseFormerV2\demo\video")
output_dir = Path(r"D:\pycharmprojects\PoseFormerV2\output_keypoints")
output_dir.mkdir(exist_ok=True)

video_paths = sorted(video_dir.glob("*.avi"))
print(f"共找到 {len(video_paths)} 个视频文件。")

for video_path in video_paths:
    print(f"\n正在处理视频: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    keypoints_all = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = simple_fullbody_bbox(frame)
        kpts = extract_keypoints(frame, bbox)
        keypoints_all.append(kpts)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f" 已处理 {frame_idx} 帧")

    cap.release()

    keypoints_np = np.stack(keypoints_all, axis=0)
    save_path = output_dir / f"{video_path.stem}_keypoints.npy"
    np.save(save_path, keypoints_np)
    print(f"保存关键点到 {save_path}，shape={keypoints_np.shape}")
