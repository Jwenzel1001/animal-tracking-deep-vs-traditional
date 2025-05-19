# Eye in the Sky: Tracking a Moving Animal Using Deep and Traditional Methods

This project compares deep learning and traditional methods for object detection and tracking in a drone-based simulated environment. The system tracks a moving animal (a gray wolf) using a drone camera in Microsoft AirSim and evaluates pipelines involving YOLO + DeepSORT and HOG+SVM + Kalman Filter.

Developed for HCI 5750: Computational Perception, Spring 2025, Iowa State University.

---

## Objectives

- Compare YOLO (deep learning) vs. HOG+SVM (traditional) for object detection
- Compare DeepSORT vs. Kalman Filter for object tracking
- Evaluate across controlled simulated scenarios:
  - Straight path
  - Zig-zag motion
  - Occlusion path
- Metrics evaluated:
  - Detection accuracy (Precision, Recall, F1 Score)
  - Tracking stability (ID Switches)
  - Computational efficiency (FPS)

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Microsoft AirSim | 3D simulation of drone and environment |
| Unreal Engine 4 | Custom map and camera control |
| Python + OpenCV | Detection, tracking, evaluation, and drone control |
| YOLOv11n (Ultralytics) | Lightweight object detection model |
| DeepSORT / Kalman Filter | Object tracking modules |

---

## Pipeline Architecture

The system follows a modular design:
1. **Drone Controller**: Aligns drone to object center using bounding box
2. **Image Capture**: Collects RGB + semantic images via AirSim API
3. **Detection**: YOLO or HOG+SVM module
4. **Tracking**: DeepSORT or Kalman Filter module
5. **Evaluation**: Computes detection/tracking metrics

Visual processing is fully scriptable with modular Python classes.

---

## Key Insights

- **YOLO** was more accurate and faster than HOG+SVM
- **Kalman Filter** outperformed DeepSORT in moving-camera scenarios due to better handling of intermittent detections
- **HOG+SVM** failed to keep up with real-time demands in a simulation, highlighting scalability issues
