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
