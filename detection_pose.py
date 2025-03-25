import cv2
import os
from ultralytics import YOLO

# === CONFIG ===
kicker_model_path = "runs/detect/kicker-detector-v2/weights/best.pt"  # Your trained kicker detector
pose_model_path = "yolov8m-pose.pt"  # Pretrained pose model
video_path = "soccer_clips/right_goal/right_angle/Copy of R(42).mp4"
output_dir = "pose_output"
os.makedirs(output_dir, exist_ok=True)

# === Load models ===
kicker_model = YOLO(kicker_model_path)
pose_model = YOLO(pose_model_path)

# === Run kicker detection with tracking (entire video) ===
results = kicker_model.track(source=video_path, stream=True, persist=True)

frame_idx = 0

for result in results:
    frame = result.orig_img
    boxes = result.boxes

    if boxes is not None and boxes.xyxy is not None:
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            # Crop kicker area from frame
            kicker_crop = frame[y1:y2, x1:x2]

            # === Step 2: Run pose estimation on the kicker crop ===
            pose_results = pose_model.predict(kicker_crop, show=False)

            for r in pose_results:
                if r.keypoints is None or len(r.keypoints.xy) == 0:
                    continue

                # === Step 3: Draw pose keypoints on the crop ===
                pose_annotated = r.plot()

                # Resize the pose-annotated crop back to its original size
                pose_resized = cv2.resize(pose_annotated, (x2 - x1, y2 - y1))

                # === Step 4: Paste back onto original frame ===
                frame[y1:y2, x1:x2] = pose_resized

    # === Step 5: Show and save output frame ===
    cv2.imshow("Kicker Pose Estimation", frame)
    cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cv2.destroyAllWindows()
print(f"✅ Done! Frames saved to: {output_dir}/")
