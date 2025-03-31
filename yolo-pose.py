from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
import csv

# Load YOLOv8 pose model
model = YOLO("yolov8m-pose.pt")

# Input folder containing clips
input_folder = "soccer_clips/right_goal/right_angle/test_angle"

# Output folder for keypoints
output_folder = "keypoints_output"
os.makedirs(output_folder, exist_ok=True)

# Loop through each .mp4 file in the folder
for video_file in os.listdir(input_folder):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(input_folder, video_file)
    base_name = os.path.splitext(video_file)[0]

    print(f"🎥 Processing {video_file}...")

    # Run pose estimation with tracking
    results = model.track(
        source=video_path,
        show=False,        # Set to True if you want to preview
        save=True,         # Save annotated video
        persist=True       # Keep tracking IDs consistent across frames
    )

    print(f"✅ Output saved to default: runs/pose/track*/track.mp4")

    # === Save keypoints to CSV ===
    output_csv = os.path.join(output_folder, f"{base_name}_keypoints.csv")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "track_id", "keypoint_x", "keypoint_y", "confidence"])  # CSV Header

        for frame_idx, result in enumerate(results):
            if result.keypoints is None:
                continue

            for i, kp in enumerate(result.keypoints.xy):  # Loop through detected players
                track_id = result.boxes.id[i] if result.boxes.id is not None else -1
                for j, (x, y) in enumerate(kp):  # Loop through keypoints
                    writer.writerow([frame_idx, track_id, x.item(), y.item(), result.keypoints.conf[i][j].item()])

    print(f"✅ Keypoints saved to {output_csv}")

print("✅ Script complete. Manually delete any videos where the striker wasn't detected.")
