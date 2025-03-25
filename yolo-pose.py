from ultralytics import YOLO
import numpy as np
import pandas as pd
import os

# Load YOLOv8 pose model
model = YOLO("yolov8m-pose.pt")

# Input folder containing clips
input_folder = "soccer_clips/right_goal/right_angle"

# Loop through each .mp4 file in the folder
for video_file in os.listdir(input_folder):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(input_folder, video_file)
    base_name = os.path.splitext(video_file)[0]

    print(f"🎥 Processing {video_file}...")

    model.track(
        source=video_path,
        show=False,        # Set to True if you want to preview
        save=True,         # Save annotated video
        persist=True       # Keep tracking IDs consistent across frames
    )

    print(f"✅ Output saved to default: runs/pose/track*/track.mp4")

print("✅ Script 1 complete. Manually delete any videos where the striker wasn't detected.")

