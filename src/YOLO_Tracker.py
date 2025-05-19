from ultralytics import YOLO
import numpy as np
import torch

class YOLODetector:
    def __init__(self, model_path=r"C:\Users\PC\Desktop\School\Graduate_School\Spring_2025\CPRE575\AirSim_Project_Code_Snippets\Wolf_Data\Vision_Pipeline\DeepLearning_Method\YOLO\YOLO_Train\runs_1\train\wolf_detector_yolov11n\weights\best.pt", class_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.class_id = class_id

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=0.3, imgsz=640, verbose=False, device=self.device.index if self.device.type == 'cuda' else 'cpu')
        detections = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0].item()) != self.class_id:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                detections.append((cx, cy, w, h, conf))

        return detections
