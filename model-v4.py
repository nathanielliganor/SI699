import os
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# === CONFIG ===
kicker_model_path = "runs/detect/kicker-detector-v2/weights/best.pt"
pose_model_path = "yolov8m-pose.pt"
video_folder = "soccer_clips/left_goal/right_angle"  # Folder containing the videos

# Output path
output_folder = "pose_data/right_goal/left_test"
os.makedirs(output_folder, exist_ok=True)

output_csv = os.path.join(output_folder, "kicker_pose_keypoints.csv")

# Smoothing factor
alpha = 0.8  # Between 0 and 1
padding = 20  # Padding around the bounding box

# === Load models ===
kicker_model = YOLO(kicker_model_path)
pose_model = YOLO(pose_model_path)

# Track smoothed bounding boxes per track ID
smoothed_boxes = defaultdict(lambda: None)

# Collect keypoint rows
pose_data = []

# === Loop through all videos in the folder ===
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
video_files.sort()  # Sort to ensure consistent ordering

for video_id, video_file in enumerate(video_files, start=1):
    video_path = os.path.join(video_folder, video_file)

    # Run kicker detection with tracking for each video
    results = kicker_model.track(source=video_path, stream=True, persist=True)

    frame_idx = 0

    for result in results:
        frame = result.orig_img
        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
            for i, box in enumerate(boxes.xyxy):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = map(int, box)

                # Smooth box per track ID
                if smoothed_boxes[track_id] is None:
                    smoothed_boxes[track_id] = [x1, y1, x2, y2]
                else:
                    prev = smoothed_boxes[track_id]
                    smoothed_boxes[track_id] = [
                        int(alpha * prev[0] + (1 - alpha) * x1),
                        int(alpha * prev[1] + (1 - alpha) * y1),
                        int(alpha * prev[2] + (1 - alpha) * x2),
                        int(alpha * prev[3] + (1 - alpha) * y2)
                    ]

                x1_s, y1_s, x2_s, y2_s = smoothed_boxes[track_id]

                # Apply padding
                x1_s = max(0, x1_s - padding)
                y1_s = max(0, y1_s - padding)
                x2_s = min(frame.shape[1], x2_s + padding)
                y2_s = min(frame.shape[0], y2_s + padding)

                # Crop the kicker
                kicker_crop = frame[y1_s:y2_s, x1_s:x2_s]

                # Run pose estimation
                pose_results = pose_model.predict(kicker_crop, show=False)

                for r in pose_results:
                    if r.keypoints is None or len(r.keypoints.xy) == 0:
                        continue

                    keypoints = r.keypoints.xy[0].cpu().numpy().flatten()  # Shape: (34,)
                    row = [frame_idx, track_id, video_id] + keypoints.tolist()  # Add video_id here
                    pose_data.append(row)

        frame_idx += 1

# === Save to CSV ===
columns = ["frame", "track_id", "video_id"] + [f"kp_{i}_{axis}" for i in range(17) for axis in ["x", "y"]]
df = pd.DataFrame(pose_data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"Keypoints saved to: {output_csv}")
