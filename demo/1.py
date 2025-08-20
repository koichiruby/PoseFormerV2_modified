import cv2

video_path = r"D:\pycharmprojects\PoseFormerV2\demo\video\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
else:
    ret, frame = cap.read()
    if ret:
        print("Video opened and read first frame successfully.")
        print("Frame shape:", frame.shape)
    else:
        print("Error: Cannot read frame.")
cap.release()
