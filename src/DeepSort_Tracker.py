# deepsort_tracker.py

import numpy as np
import sys
import os

deep_sort_path = os.path.join(
    os.getcwd(), 
    "DeepLearning_Method", 
    "DeepSORT", 
    "deep_sort_pytorch"
)
sys.path.append(deep_sort_path)

from deep_sort.deep_sort import DeepSort

class DeepSortTracker:
    def __init__(self, model_path=r"C:\Users\PC\Desktop\School\Graduate_School\Spring_2025\CPRE575\AirSim_Project_Code_Snippets\Wolf_Data\Vision_Pipeline\DeepLearning_Method\DeepSORT\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7"):
        self.deepsort = DeepSort(model_path=model_path, max_dist=0.8)

    def update(self, detections, frame):
        if not detections:
            return np.empty((0, 6))

        bbox_xywh = []
        confs = []
        classes = []

        for det in detections:
            cx, cy, w, h, conf = det
            bbox_xywh.append([cx, cy, w, h])
            confs.append(conf)
            classes.append(0)

        bbox_xywh = np.array(bbox_xywh)
        confs = np.array(confs)
        classes = np.array(classes)

        # Returns corner format
        outputs, _ = self.deepsort.update(bbox_xywh, confs, classes, frame)

        # Convert corner format back to center-based format
        results = []
        for x1, y1, x2, y2, cls_id, track_id in outputs:
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            results.append((cx, cy, w, h, track_id))

        return np.array(results)
