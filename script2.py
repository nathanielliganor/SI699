import os
from ultralytics import YOLO
import pandas as pd
import cv2

# === Setup ===
model = YOLO("yolov8m-pose.pt")
input_folder = "soccer_clips/right_goal/right_angle"
output_folder = "pose_data/right_goal/right_angle"
os.makedirs(output_folder, exist_ok=True)

# === Loop through each video ===
for video_file in sorted(os.listdir(input_folder)):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(input_folder, video_file)
    clip_name = os.path.splitext(video_file)[0]
    print(f"\n🎥 Now processing: {video_file}")

    # === Step 1: Run tracking with stream=True and persist IDs ===
    results = list(model.track(source=video_path, stream=True, show=True, persist=True))

    if len(results) < 2:
        print(f"⚠️ Not enough frames in {video_file} to display a 75% frame.")
        continue

    # === Step 2: Manually preview several frames near the end ===
    start_idx = int(len(results) * 0.7)
    end_idx = int(len(results) * 0.95)

    print("🕵️ Previewing frames near the end of the video (press any key to advance)...")
    for i in range(start_idx, end_idx):
        im = results[i].plot()
        cv2.imshow(f"Striker Selection: {clip_name}", im)
        key = cv2.waitKey(0)
        if key == ord("s"):  # Press 's' to stop and choose this frame
            break
    cv2.destroyAllWindows()

    # Prompt for striker ID
    try:
        striker_id = int(input(f"👤 Enter the striker's track ID for '{video_file}': "))
    except ValueError:
        print("❌ Invalid input. Skipping this video.")
        continue

    # === Step 3: Extract striker keypoints ===
    print(f"📌 Extracting keypoints for striker ID {striker_id}...")
    pose_data = []
    for frame_num, result in enumerate(results):
        if result.keypoints is None or result.boxes is None:
            continue

        ids = result.boxes.id
        keypoints = result.keypoints.xy

        if ids is None:
            continue

        for i, person_id in enumerate(ids):
            if int(person_id) == striker_id:
                kp = keypoints[i].cpu().numpy().flatten()
                pose_data.append([frame_num] + kp.tolist())
                break

    # === Step 4: Save keypoints to CSV ===
    if pose_data:
        columns = ["frame"] + [f"kp_{i}_{axis}" for i in range(17) for axis in ["x", "y"]]
        df = pd.DataFrame(pose_data, columns=columns)
        csv_path = os.path.join(output_folder, f"{clip_name}_pose.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Keypoints saved to {csv_path}")
    else:
        print(f"⚠️ No keypoints found for striker ID {striker_id} in {video_file}.")

print("\n🎉 All videos processed!")
