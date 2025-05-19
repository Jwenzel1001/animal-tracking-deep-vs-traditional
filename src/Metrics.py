import time
import numpy as np
import cv2

class MetricsRecorder:
    def __init__(self):
        self.start_time = None
        self.frame_times = []

        self.correct_detections = 0
        self.false_positives = 0
        self.total_detections = 0
        self.total_ground_truths = 0

        self.last_wolf_id = None
        self.id_switches = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        elapsed = time.time() - self.start_time
        self.frame_times.append(elapsed)

    def update_detection_accuracy(self, semantic_img, detections, wolf_color_bgr=(106, 31, 92), iou_threshold=0.5):
        mask = cv2.inRange(semantic_img, wolf_color_bgr, wolf_color_bgr)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return

        self.total_ground_truths += 1
        self.total_detections += len(detections)

        x, y, w, h = cv2.boundingRect(np.array(list(zip(xs, ys))))
        gt_bbox = np.array([x, y, x + w, y + h])

        matched = False
        for det in detections:
            cx, cy, dw, dh, _ = det
            dx1 = cx - dw / 2
            dy1 = cy - dh / 2
            dx2 = cx + dw / 2
            dy2 = cy + dh / 2
            det_bbox = np.array([dx1, dy1, dx2, dy2])

            iou = self.compute_iou(gt_bbox, det_bbox)
            if iou >= iou_threshold:
                self.correct_detections += 1
                matched = True
                break

        if not matched and len(detections) > 0:
            self.false_positives += len(detections)

    def update_id_switches(self, outputs, semantic_img, wolf_color_bgr=(106, 31, 92), iou_threshold=0.5):
        mask = cv2.inRange(semantic_img, wolf_color_bgr, wolf_color_bgr)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return

        x, y, w, h = cv2.boundingRect(np.array(list(zip(xs, ys))))
        gt_bbox = np.array([x, y, x + w, y + h])

        for output in outputs:
            cx, cy, bw, bh, track_id = output
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            pred_box = np.array([x1, y1, x2, y2])

            iou = self.compute_iou(gt_bbox, pred_box)
            if iou >= iou_threshold:
                if self.last_wolf_id is not None and track_id != self.last_wolf_id:
                    self.id_switches += 1
                self.last_wolf_id = track_id
                break

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / (boxAArea + boxBArea - interArea)

    def compute_metrics(self):
        avg_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0.0
        precision = (self.correct_detections / self.total_detections) * 100 if self.total_detections > 0 else 0.0
        recall = (self.correct_detections / self.total_ground_truths) * 100 if self.total_ground_truths > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return avg_fps, precision, recall, f1, self.false_positives, self.id_switches

    def print_summary(self, output_path="metrics_summary.txt"):
        avg_fps, precision, recall, f1, false_positives, id_switches = self.compute_metrics()
        summary = (
            "\n==== Final Metrics Summary ====\n"
            f"Average FPS: {avg_fps:.2f}\n"
            f"Precision: {precision:.2f}%\n"
            f"Recall (Detection Accuracy): {recall:.2f}%\n"
            f"F1 Score: {f1:.2f}%\n"
            f"False Positives: {false_positives}\n"
            f"Total ID Switches (wolf only): {id_switches}\n"
            f"Total Frames: {len(self.frame_times)}\n"
            f"Correct Detections: {self.correct_detections}\n"
            f"Total Detections: {self.total_detections}\n"
            f"Total Ground Truths: {self.total_ground_truths}\n"
            "================================\n"
        )
        print(summary)
        with open(output_path, "w") as f:
            f.write(summary)