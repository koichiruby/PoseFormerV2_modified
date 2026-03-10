Trampoline 3D Pose Estimation & Action Quality Assessment
<p align="center">

Single-view 3D human pose reconstruction and action scoring for trampoline sports
based on PoseFormerV2 and Transformer temporal modeling.

</p> <p align="center">










</p>
📖 Project Overview

Trampoline routines involve high-speed rotation, large vertical displacement,
and complex body dynamics, making manual motion analysis difficult.

This project builds an end-to-end computer vision pipeline that reconstructs 3D human skeletons from single-camera videos and performs automatic action quality assessment (AQA).

The framework integrates:

2D keypoint detection

3D pose reconstruction

temporal motion modeling

trajectory visualization

automatic scoring

It provides a tool for objective motion analysis in trampoline training and sports science research.

🚀 Pipeline

The full pipeline of the proposed system:

Video Input
     │
     ▼
YOLOv8-Pose 2D Keypoint Detection
     │
     ▼
Main Athlete Selection
     │
     ▼
Temporal Smoothing
(Savitzky–Golay Filter)
     │
     ▼
PoseFormerV2
2D → 3D Pose Reconstruction
     │
     ▼
Post-Processing
• Bone Length Constraint
• Temporal Stabilization
• Coordinate Normalization
     │
     ▼
3D Skeleton Visualization
     │
     ▼
Action Quality Assessment
(Regression Model)
🧠 Key Features
✔ Robust 2D Pose Detection

YOLOv8-Pose based keypoint detection

Main athlete tracking

Temporal smoothing

Missing frame compensation

✔ Transformer-based 3D Reconstruction

Uses PoseFormerV2 to model:

spatial joint relationships

long-term temporal dependencies

Input:

(T × 17 × 2)

Output:

(T × 17 × 3)
✔ Motion Stabilization

To handle high-speed trampoline motion, the system applies:

Savitzky–Golay temporal filtering

bone length constraint

world coordinate transformation

These steps reduce skeleton jitter and unrealistic deformation.

✔ Automatic Action Scoring

The reconstructed 3D skeleton sequence is used to train a regression network for action quality assessment.

Evaluation metrics:

MAE

RMSE

Spearman Rank Correlation

📊 Experimental Results
3D Pose Estimation
Metric	Value
MPJPE	46 mm
P-MPJPE	37 mm

MPJPE measures the average Euclidean distance between predicted and ground truth joint positions.

Action Quality Assessment
Metric	Value
MAE	4.46
RMSE	6.17
Spearman	0.66

Results show that the model can predict action scores and maintain ranking consistency with ground-truth evaluations.

🎥 Visualization

The system supports:

2D skeleton overlay on video

3D skeleton reconstruction

motion trajectory visualization

side-by-side comparison

Example workflow:

Video Frame → 2D Pose → 3D Skeleton → Motion Trajectory

This allows intuitive observation of:

jump height

rotation phase

posture stability

📂 Project Structure
PoseFormerV2-Trampoline/
│
├── demo/
│   └── npz.py
│
├── data/
│   └── train_output/
│
├── inference/
│   └── filtered_keypoints_interp/
│
├── models/
│
├── visualization/
│
├── checkpoints/
│
└── README.md
⚙ Installation

Clone the repository:

git clone https://github.com/yourname/Trampoline-Action-Recognition-PoseFormerV2.git

cd Trampoline-Action-Recognition-PoseFormerV2

Install dependencies:

pip install -r requirements.txt
▶ Run Demo

Run the demo script:

python demo/npz.py

The script will:

load 2D keypoints

run PoseFormerV2 inference

generate 3D pose visualization

📦 Dataset

This project uses:

AQA-7

Trampoline action quality assessment dataset

FineDiving

Used for model pretraining and motion representation learning

⚠ Limitations

Since the system relies on single-view video, depth ambiguity may occur during:

high-speed rotations

self-occlusion

trampoline contact phase

Future work may incorporate multi-view cameras or depth sensors.

🔮 Future Work

Possible improvements include:

multi-view 3D pose estimation

transformer-based scoring models

larger trampoline motion datasets

real-time training assistance systems

📚 Citation

If you find this project useful, please cite:

PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation
CVPR 2023
