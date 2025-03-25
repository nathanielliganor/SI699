from ultralytics import YOLO
import numpy as np
import pandas as pd
import os

# # Load the pose model
# model = YOLO("yolov8m-pose.pt")

# # Define your video source and striker ID
# video_path = "soccer_clips/right_goal/right_angle/Copy of R.mp4"
# striker_id = 3 

# # Run tracking with streaming
# results = model.track(
#     source=video_path,
#     show=True,           # Show the video while processing
#     save=True,           # Save the annotated video
#     persist=True         # Keep track_ids consistent
# )

# pose_data = []
# frame_num = 0

# for result in results:
#     if result.keypoints is None or result.boxes is None:
#         frame_num += 1
#         continue

#     # Get tracking IDs and keypoints
#     ids = result.boxes.id
#     keypoints = result.keypoints.xy  # Shape: [num_people, 17, 2]

#     if ids is None:
#         frame_num += 1
#         continue

#     for i, person_id in enumerate(ids):
#         if int(person_id) == striker_id:
#             kp = keypoints[i].cpu().numpy().flatten()  # shape: (34,) -> x0,y0,x1,y1,...x16,y16
#             pose_data.append([frame_num] + kp.tolist())
#             break  # No need to check other people this frame

#     frame_num += 1

# # Convert to DataFrame
# columns = ["frame"]
# for i in range(17):
#     columns += [f"kp_{i}_x", f"kp_{i}_y"]

# df = pd.DataFrame(pose_data, columns=columns)

# # Save to CSV
# df.to_csv("striker_pose.csv", index=False)

# print("Pose keypoints saved to striker_pose.csv")

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

