Trampoline Action Analysis Based on PoseFormerV2

A computer vision pipeline for 3D human pose reconstruction and action quality assessment (AQA) from single-camera trampoline videos using PoseFormerV2.

This project reconstructs 3D skeleton sequences from 2D keypoints and performs automatic action scoring based on the reconstructed motion features.

Project Overview

Trampoline sports involve high-speed rotation, large vertical displacement, and complex body movements, making manual motion analysis difficult.

This project builds an end-to-end pipeline that:

Extracts 2D human keypoints from trampoline videos

Reconstructs 3D skeleton sequences using PoseFormerV2

Applies temporal smoothing and skeleton constraints

Generates 3D trajectory visualization

Performs automatic action quality assessment

The system can assist coaches and athletes in analyzing motion patterns and evaluating performance objectively.

Pipeline

The full workflow of the system:

Video Input
     │
     ▼
YOLOv8-Pose 2D Keypoint Detection
     │
     ▼
Main Athlete Selection
     │
     ▼
Temporal Smoothing (Savitzky–Golay)
     │
     ▼
PoseFormerV2 2D→3D Pose Reconstruction
     │
     ▼
Post-processing
  • Temporal smoothing
  • Bone length constraint
  • Coordinate normalization
     │
     ▼
3D Skeleton Visualization
     │
     ▼
Action Quality Assessment (Regression Model)
Example Result

The system produces synchronized 2D pose detection and 3D pose reconstruction.

Example:

Left: 2D skeleton overlay on trampoline video

Right: reconstructed 3D skeleton with trajectory

This allows intuitive observation of motion patterns, rotation phases, and trajectory changes.

Model Performance
3D Pose Estimation
Metric	Value
MPJPE	46 mm
P-MPJPE	37 mm

MPJPE measures the average Euclidean distance between predicted and ground-truth joint positions.

Action Quality Assessment
Metric	Value
MAE	4.46
RMSE	6.17
Spearman	0.66

The results show that the model can predict action scores and maintain ranking consistency with ground truth evaluations.

Dataset

This project uses two datasets:

AQA-7

Used for trampoline action quality assessment

FineDiving

Used for pretraining and improving motion representation

Key Methods
2D Pose Detection

YOLOv8-Pose is used to extract 17 human keypoints.

Additional strategies are applied:

main athlete selection

temporal keypoint smoothing

missing frame compensation

3D Pose Reconstruction

The project uses PoseFormerV2, a transformer-based model that models:

Spatial relationships between joints

Temporal dependencies across frames

Input:

(T × 17 × 2)

Output:

(T × 17 × 3)
Post-processing

To improve stability in high-speed trampoline movements:

• Savitzky–Golay temporal filtering
• Bone-length constraint
• World coordinate transformation

These steps reduce jitter and unrealistic skeleton deformation.

Action Quality Assessment

3D skeleton sequences are fed into a regression network that predicts trampoline action scores.

Evaluation metrics include:

MAE

RMSE

Spearman Rank Correlation

Visualization

The system provides:

2D skeleton overlay

3D skeleton reconstruction

3D motion trajectory

side-by-side comparison video

This helps analyze jump height, rotation phase, and posture stability.

Installation
git clone https://github.com/yourname/Trampoline-Action-Recognition-PoseFormerV2.git

cd Trampoline-Action-Recognition-PoseFormerV2

pip install -r requirements.txt
Run Demo
python demo/npz.py

This will:

Load 2D keypoints

Run PoseFormerV2 inference

Generate 3D skeleton visualization

Project Structure
PoseFormerV2-Trampoline/
│
├── demo/
│   ├── npz.py
│
├── data/
│   ├── train_output/
│
├── inference/
│   ├── filtered_keypoints_interp/
│
├── models/
│
├── visualization/
│
└── README.md
Limitations

Because the system uses single-camera video, depth estimation errors may occur in:

high-speed rotation

body occlusion

trampoline contact phases

Future work could integrate multi-view cameras or depth sensors.

Future Work

Possible improvements:

multi-view 3D pose reconstruction

transformer-based scoring networks

larger trampoline motion datasets

real-time motion analysis systems

Citation

If you find this project helpful, please consider citing:

PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation
CVPR 2023
Author

Yuxi Deng
Beijing Sport University
Data Science and Big Data Technology
