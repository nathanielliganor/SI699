import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# === CONFIG ===
kicker_model_path = "runs/detect/kicker-detector-v2/weights/best.pt"
pose_model_path = "yolov8m-pose.pt"
video_path = "soccer_clips/right_goal/left_angle/Copy of L(23).mp4"
output_dir = "pose_output_smoothed"
os.makedirs(output_dir, exist_ok=True)

# Smoothing factor (between 0 and 1)
alpha = 0.8  # Higher = less smoothing, lower = more stable

# === Load models ===
kicker_model = YOLO(kicker_model_path)
pose_model = YOLO(pose_model_path)

# === Initialize smoothing tracker ===
smoothed_boxes = defaultdict(lambda: None)

# === Run kicker detection with tracking ===
results = kicker_model.track(source=video_path, stream=True, persist=True)

frame_idx = 0

for result in results:
    frame = result.orig_img
    boxes = result.boxes

    if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
        for i, box in enumerate(boxes.xyxy):
            track_id = int(boxes.id[i])  # Use tracking ID
            x1, y1, x2, y2 = map(int, box)

            # Initialize or smooth box
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

            # Use smoothed box
            x1_s, y1_s, x2_s, y2_s = smoothed_boxes[track_id]

            # Optional padding
            padding = 20
            x1_s = max(0, x1_s - padding)
            y1_s = max(0, y1_s - padding)
            x2_s = min(frame.shape[1], x2_s + padding)
            y2_s = min(frame.shape[0], y2_s + padding)

            kicker_crop = frame[y1_s:y2_s, x1_s:x2_s]

            # === Pose Estimation on the cropped kicker ===
            pose_results = pose_model.predict(kicker_crop, show=False)

            for r in pose_results:
                if r.keypoints is None or len(r.keypoints.xy) == 0:
                    continue

                pose_annotated = r.plot()
                pose_resized = cv2.resize(pose_annotated, (x2_s - x1_s, y2_s - y1_s))
                frame[y1_s:y2_s, x1_s:x2_s] = pose_resized

    # === Save & Show ===
    cv2.imshow("Smoothed Kicker Pose", frame)
    cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cv2.destroyAllWindows()
print(f"✅ Done! Smoothed pose-enhanced frames saved to: {output_dir}/")
