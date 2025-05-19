import cv2
import numpy as np
from skimage.feature import hog
from imutils.object_detection import non_max_suppression
import joblib

class HOGSVMDetector:
    def __init__(self, model_path, window_size=(128, 128), step_size=32,
                 scale_factor=1.25, score_threshold=2):
        self.model = joblib.load(model_path)
        self.window_size = window_size
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.score_threshold = score_threshold
        self.hog_params = {
            'pixels_per_cell': (8, 8),
            'cells_per_block': (3, 3),
            'orientations': 9,
            'block_norm': 'L2-Hys'
        }

    # Apply image pyramid onto window for detections if the wolf sizes varies
    def pyramid(self, image, scale=1.25, min_size=(64, 144)):
        current_scale = 1.0
        yield image, current_scale
        while True:
            w = int(image.shape[1] / scale)
            h = int(image.shape[0] / scale)
            image = cv2.resize(image, (w, h))
            current_scale *= scale
            if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
                break
            yield image, current_scale

    # Sliding window to stride over input image
    def sliding_window(self, image):
        for y in range(0, image.shape[0] - self.window_size[1], self.step_size):
            for x in range(0, image.shape[1] - self.window_size[0], self.step_size):
                yield (x, y, image[y:y + self.window_size[1], x:x + self.window_size[0]])

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        # Performing detection with image pyramid and sliding windows
        for resized, current_scale in self.pyramid(gray, self.scale_factor):
            for (x, y, window) in self.sliding_window(resized):
                if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
                    continue
                features = hog(window, **self.hog_params)
                score = self.model.decision_function([features])[0]
                if score > self.score_threshold:
                    x1 = int(x * current_scale)
                    y1 = int(y * current_scale)
                    x2 = int((x + self.window_size[0]) * current_scale)
                    y2 = int((y + self.window_size[1]) * current_scale)
                    detections.append((x1, y1, x2, y2, score))

        # Apply Non-Max Suppression
        if detections:
            boxes = np.array([d[:4] for d in detections])
            scores = np.array([d[4] for d in detections])
            picks = non_max_suppression(boxes, probs=scores, overlapThresh=0.3)
        else:
            picks = []

        # Convert to (cx, cy, w, h, score) format
        formatted = []
        for (x1, y1, x2, y2) in picks:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            score = next((s for (bx1, by1, bx2, by2, s) in detections if (x1, y1, x2, y2) == (bx1, by1, bx2, by2)), 0)
            formatted.append((cx, cy, w, h, score))

        return formatted
