from ultralytics import YOLO

# === CONFIGURATION ===
model_path = "runs/detect/kicker-detector-v2/weights/best.pt"  # Path to your trained model
video_path = "soccer_clips/right_goal/right_angle/Copy of R(42).mp4"  # Path to the video you want to test
save_output = True  # Set to True if you want to save the output video
show_output = True  # Set to True to display video while processing

# === LOAD MODEL ===
model = YOLO(model_path)

# === RUN DETECTION ===
results = model.track(
    source=video_path,
    show=show_output,
    save=save_output,
    project="test_outputs",
    name="kicker_demo",
    persist=True  # Ensures track IDs are consistent across frames
)

print("✅ Done! Check output folder: test_outputs/kicker_demo/")
