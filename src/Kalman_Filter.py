import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        cx, cy, w, h = bbox
        measurement = np.array([cx, cy, w, h], dtype=np.float32)

        self.kf.statePre = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePre[0:4, 0] = measurement
        self.kf.statePost[0:4, 0] = measurement
        self.kf.statePre[4:8, 0] = 1e-2
        self.kf.statePost[4:8, 0] = 1e-2
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        print(f"[Tracker {self.id}] Initialized with bbox: cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}")

    def update(self, bbox):
        self.kf.correct(np.array(bbox, dtype=np.float32))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        print(f"[Tracker {self.id}] Corrected with: {bbox}")
        print(f"[Tracker {self.id}] Post-correction state: {self.kf.statePost.ravel()}")

    def predict(self):
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        print(f"[Tracker {self.id}] Predicted state: {prediction.ravel()}")
        return prediction[:4, 0]

    def get_state(self):
        return self.kf.statePost[:4, 0]


class SimpleKalmanTracker:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections):
        print(f"\n[SimpleKalmanTracker] Raw detections: {detections}")
        dets_center = np.array(detections, dtype=np.float32)
        print(f"[SimpleKalmanTracker] Center detections: {dets_center}")

        trks = np.array([trk.predict() for trk in self.trackers], dtype=np.float32)
        print(f"[SimpleKalmanTracker] Tracker predictions: {trks}")

        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets_center, trks)
        print(f"[SimpleKalmanTracker] Matches: {matches}")

        for t_idx, trk in enumerate(self.trackers):
            if t_idx in matches:
                d_idx = matches[t_idx]
                trk.update(dets_center[d_idx, :4])

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets_center[d, :4]))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        outputs = []
        for trk in self.trackers:
            if trk.hit_streak >= 2 and trk.time_since_update <= self.max_age:
                cx, cy, w, h = trk.get_state()
                outputs.append([cx, cy, w, h, trk.id])

        return np.array(outputs) if outputs else np.empty((0, 5))

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return {}, list(range(len(dets))), []
        if len(dets) == 0:
            return {}, [], list(range(len(trks)))

        iou_mat = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_mat[d, t] = self.iou(det[:4], trk)

        matched_indices = linear_sum_assignment(-iou_mat)
        matched = list(zip(*matched_indices))
        matches = {}
        unmatched_dets = []
        unmatched_trks = []

        for d, t in matched:
            if iou_mat[d, t] >= self.iou_threshold:
                matches[t] = d
            else:
                unmatched_dets.append(d)
                unmatched_trks.append(t)

        unmatched_dets += [d for d in range(len(dets)) if d not in [m[0] for m in matched]]
        unmatched_trks += [t for t in range(len(trks)) if t not in [m[1] for m in matched]]

        return matches, unmatched_dets, unmatched_trks

    @staticmethod
    def iou(b1, b2):
        cx1, cy1, w1, h1 = b1
        cx2, cy2, w2, h2 = b2
        x1min, y1min = cx1 - w1 / 2, cy1 - h1 / 2
        x1max, y1max = cx1 + w1 / 2, cy1 + h1 / 2
        x2min, y2min = cx2 - w2 / 2, cy2 - h2 / 2
        x2max, y2max = cx2 + w2 / 2, cy2 + h2 / 2

        xx1 = max(x1min, x2min)
        yy1 = max(y1min, y2min)
        xx2 = min(x1max, x2max)
        yy2 = min(y1max, y2max)

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        union = w1 * h1 + w2 * h2 - inter

        return inter / union if union > 0 else 0.0