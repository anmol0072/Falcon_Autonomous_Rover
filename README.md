1. Install Anaconda
2. create an envirnment for this project

REQUIERMENTS:
1. Pytorch
2. OpenCV
3. Scikit.learn
4. transforms
5. Datetime
6. tqmd


 # 🦅 Falcon Autonomous Rover

**A robust autonomous navigation system using SWA-enhanced Semantic Segmentation.**
*Built for [Hackathon Name]*

![Project Status](https://img.shields.io/badge/Status-Prototype-green)
![Model](https://img.shields.io/badge/Arch-UNet%20MobileNet-blue)
![Technique](https://img.shields.io/badge/Training-SWA-orange)

## 📖 Overview
The Falcon Rover is an autonomous edge-robot designed to navigate unstructured Martian terrain. It uses a **UNet architecture with a MobileNet backbone** to process visual data in real-time. 

To ensure high reliability and better generalization across different lighting conditions, we utilized **Stochastic Weight Averaging (SWA)** during the training process.

---

## 🚀 Key Features
* **Architecture:** UNet (MobileNetV3 Encoder) for lightweight, high-speed segmentation.
* **Training Technique:** SWA (Stochastic Weight Averaging) to create a more robust model that generalizes better than standard training.
* **Efficiency:** Optimized for edge inference on Mac M1 (MPS) and Standard CPUs.
* **Performance:** ~35 FPS Inference speed with stable decision boundaries.

---

## ⚙️ Requirements

### Hardware
* **Laptop/Edge Device:** NVIDIA GPU, Apple Silicon (Mac M1/M2/M3), or Modern CPU.


### Software Dependencies
```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib tqdm


🛠️ How to Run
1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/Falcon_Autonomous_Rover.git](https://github.com/YOUR_USERNAME/Falcon_Autonomous_Rover.git)
cd Falcon_Autonomous_Rover

Component,Choice,Reason
Model,UNet,Excellent for localized boundary detection (finding road edges).
Backbone,MobileNetV3,Ultra-lightweight for real-time FPS on laptops.
Optimizer,SWA,Averaging weights leads to flatter minima and better generalization.

Training Configuration
Epochs: 15 (Converged stability)

Input Size: 320x320 RGB

Classes: 4 (Sky, Sand/Road, Obstacle, Vegetation)

Optimization: Stochastic Weight Averaging enabled in final epochs.


📊 How to Reproduce Results
To verify our training pipeline:

Prepare Data: Ensure synthetic dataset is in dataset/train.

Run Pre-processing:

Bash
python prepare_data.py




Train the Model:

Bash
python train_unet.py


Evaluate:

Bash
python evaluate_unet.py

Color,Class,Robot Behavior
🟨 Yellow,Road (Sand),Target. The robot steers to center this mass.
🟥 Red,Obstacle,Avoid. Triggers emergency stop or turn.
🟩 Green,Vegetation,Avoid. Treated as non-drivable.
⬛ Black,Sky,Ignored.

Here is the complete **README.md** file content. Click "Copy" on the top right of the code block and paste it directly into your GitHub file.

```markdown
# 🦅 Falcon Autonomous Rover

**A robust autonomous navigation system using SWA-enhanced Semantic Segmentation.**
*Built for [Hackathon Name]*

![Project Status](https://img.shields.io/badge/Status-Prototype-green)
![Model](https://img.shields.io/badge/Arch-UNet%20MobileNet-blue)
![Technique](https://img.shields.io/badge/Training-SWA-orange)

## 📖 Overview
The Falcon Rover is an autonomous edge-robot designed to navigate unstructured Martian terrain. It uses a **UNet architecture with a MobileNet backbone** to process visual data in real-time. 

To ensure high reliability and better generalization across different lighting conditions, we utilized **Stochastic Weight Averaging (SWA)** during the training process.

---

## 🚀 Key Features
* **Architecture:** UNet (MobileNetV3 Encoder) for lightweight, high-speed segmentation.
* **Training Technique:** SWA (Stochastic Weight Averaging) to create a more robust model that generalizes better than standard training.
* **Efficiency:** Optimized for edge inference on Mac M1 (MPS) and Standard CPUs.
* **Performance:** ~35 FPS Inference speed with stable decision boundaries.

---

## ⚙️ Requirements

### Hardware
* **Laptop/Edge Device:** NVIDIA GPU, Apple Silicon (Mac M1/M2/M3), or Modern CPU.
* **Camera:** Standard Webcam (USB or Built-in).

### Software Dependencies
```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib tqdm

```

---

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/Falcon_Autonomous_Rover.git](https://github.com/YOUR_USERNAME/Falcon_Autonomous_Rover.git)
cd Falcon_Autonomous_Rover

```

### 2. Launch the Autonomous Pilot

This script loads the SWA-optimized model and starts the decision engine.

```bash
python drive.py

```

**Controls:**

* `Q`: Quit the rover.
* **Display:** The window shows a split view: [Raw Camera Feed] | [AI Segmentation Mask].

---

## 🧠 Model & Training Details

We prioritized **generalization** over raw training accuracy to handle the "Sim-to-Real" gap effectively.

### Architecture Choice

| Component | Choice | Reason |
| --- | --- | --- |
| **Model** | **UNet** | Excellent for localized boundary detection (finding road edges). |
| **Backbone** | **MobileNetV3** | Ultra-lightweight for real-time FPS on laptops. |
| **Optimizer** | **SWA** | Averaging weights leads to flatter minima and better generalization. |

### Training Configuration

* **Epochs:** 15 (Converged stability)
* **Input Size:** 320x320 RGB
* **Classes:** 4 (Sky, Sand/Road, Obstacle, Vegetation)
* **Optimization:** Stochastic Weight Averaging enabled in final epochs.

---

## 📊 How to Reproduce Results

To verify our training pipeline:

1. **Prepare Data:** Ensure synthetic dataset is in `dataset/train`.
2. **Run Pre-processing:**
```bash
python prepare_data.py

```


3. **Train the Model:**
```bash
python train.py

```


*This will run for 15 epochs and save the SWA-averaged weights to `best_model.pth`.*
4. **Evaluate:**
```bash
python evaluate.py

```



---

## 🌈 Interpreting the Vision System

The AI segments the world into 4 color-coded classes:

| Color | Class | Robot Behavior |
| --- | --- | --- |
| 🟨 **Yellow** | **Road (Sand)** | **Target.** The robot steers to center this mass. |
| 🟥 **Red** | **Obstacle** | **Avoid.** Triggers emergency stop or turn. |
| 🟩 **Green** | **Vegetation** | **Avoid.** Treated as non-drivable. |
| ⬛ **Black** | **Sky** | Ignored. |

---

## ⚠️ Notes & Troubleshooting

### 1. "Black Screen" on Mac

If the camera feed is black, ensure Terminal has **Camera Permissions** in System Settings.

### 2. SWA Weights

The `best_model.pth` contains the averaged weights. If loading fails with strict key errors, use `strict=False` in PyTorch (handled automatically in our `drive.py`).

### 3. Unexpected Outputs

If the robot detects a carpet as "Road", this is due to the domain gap between synthetic training data and the real world. For best demo results, use a plain surface (bedsheet or table) to simulate the texture of Martian sand.

```

```
