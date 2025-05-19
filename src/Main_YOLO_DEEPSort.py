# Main_DeepSort.py

import airsim
import time
import cv2
import numpy as np
from YOLO_Tracker import YOLODetector
from DeepSort_Tracker import DeepSortTracker
from Drone_Controller import DroneController
from Metrics import MetricsRecorder

# === Connect to AirSim ===
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-6.0, 2).join()

# === Assign segmentation IDs ===
client.simSetSegmentationObjectID(".*", 1, True)
client.simSetSegmentationObjectID("BP_Spline_Wolf_17", 42, True)

# === Initialize Modules ===
detector = YOLODetector(
    model_path=r"C:\Users\PC\Desktop\School\Graduate_School\Spring_2025\CPRE575\AirSim_Project_Code_Snippets\Wolf_Data\Vision_Pipeline\DeepLearning_Method\YOLO\YOLO_Train\runs_1\train\wolf_detector_yolov11n\weights\best.pt",
    class_id=0
)
tracker = DeepSortTracker(
    model_path=r"C:\Users\PC\Desktop\School\Graduate_School\Spring_2025\CPRE575\AirSim_Project_Code_Snippets\Wolf_Data\Vision_Pipeline\DeepLearning_Method\DeepSORT\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7"
)
controller = DroneController(client)
metrics = MetricsRecorder()

# === Helper: Fetch RGB + Semantic Image ===
def get_airsim_images(client):
    responses = client.simGetImages([
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("3", airsim.ImageType.Segmentation, False, False)
    ])
    rgb_resp, semantic_resp = responses

    if rgb_resp.height == 0 or semantic_resp.height == 0:
        return None, None

    rgb1d = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
    rgb_img = rgb1d.reshape(rgb_resp.height, rgb_resp.width, 3).copy()

    semantic1d = np.frombuffer(semantic_resp.image_data_uint8, dtype=np.uint8)
    semantic_img = semantic1d.reshape(semantic_resp.height, semantic_resp.width, 3).copy()

    return rgb_img, semantic_img

# === Main Loop ===
frame_idx = 0

while True:
    metrics.start_timer()

    frame, semantic_frame = get_airsim_images(client)
    if frame is None or semantic_frame is None:
        print("Warning: No frame received!")
        continue

    # Run YOLO detection every specified frames
    if frame_idx % 1 == 0:
        detections = detector.detect(frame)
        print(f"[Frame {frame_idx}] YOLO detections: {len(detections)}")
    else:
        detections = []

    detections.sort(key=lambda x: x[4], reverse=True)

    # Update DeepSORT
    outputs = tracker.update(detections, frame)
    print(f"[Frame {frame_idx}] Tracking outputs: {len(outputs)}")

    # === Update Metrics ===
    metrics.stop_timer()

    if frame_idx % 1 == 0:
        metrics.update_detection_accuracy(semantic_frame, detections)

    metrics.update_id_switches(outputs, semantic_frame)

    # === Visualize Tracking Results ===
    if len(outputs) > 0:
        for output in outputs:
            cx, cy, w, h, track_id = output
            # Convert to corner-based for display
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # === Control Drone ===
        cx, cy, w, h, track_id = outputs[0] # Outputs are center-based
        controller.update((cx, cy), frame.shape)

    # === Display the frame ===
    cv2.imshow('AirSim Tracking View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1
    time.sleep(0.1)  # 10Hz loop

# === After the Simulation ===
cv2.destroyAllWindows()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# === Print final metrics ===
metrics.print_summary("Path_2.txt")
