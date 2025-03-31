import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths to both CSV files
right_csv_path = "pose_data/right_goal/right_test/kicker_pose_keypoints.csv"
left_csv_path = "pose_data/left_goal/left_test/kicker_pose_keypoints.csv"

# Load both datasets
df_right = pd.read_csv(right_csv_path)
df_left = pd.read_csv(left_csv_path)

# Modify the video_id to distinguish left and right sides
df_right["video_id"] = df_right["video_id"].astype(str) + "r"
df_left["video_id"] = df_left["video_id"].astype(str) + "l"

# Assign a "side" column to differentiate data
df_right["side"] = "right"
df_left["side"] = "left"


# Combine both CSVs
df = pd.concat([df_right, df_left], ignore_index=True)


# Ensure data is sorted by video_id and frame number
df = df.sort_values(by=["video_id", "frame"]).reset_index(drop=True)

print(f"Total videos found: {df['video_id'].nunique()}")  # Should match all videos from both angles

# Define the number of frames to standardize
NUM_FRAMES = 30

# Extract unique video IDs
video_ids = df["video_id"].unique()
keypoint_cols = [col for col in df.columns if "kp_" in col]
zero_percentage = (df[keypoint_cols] == 0).mean()  # True = 1, False = 0, so .mean() gives percentage of zeros
# Filter out columns where more than 25% of the values are zeros
columns_to_use = zero_percentage[zero_percentage <= 0.50].index.tolist()
keypoint_cols = [col for col in columns_to_use]



# Store processed sequences
video_sequences = []
video_sides = []  # To store the side (left or right) for each sequence

for vid in video_ids:
    vid_data = df[df["video_id"] == vid]

    # Select the last 40 frames
    vid_data = vid_data.iloc[-NUM_FRAMES:] if len(vid_data) >= NUM_FRAMES else vid_data

    # Pad if the video has fewer than 40 frames
    while len(vid_data) < NUM_FRAMES:
        vid_data = pd.concat([vid_data, vid_data.iloc[-1:]])  # Repeat last row

    # Keep only keypoints
    video_sequences.append(vid_data[keypoint_cols].values)

    # Store side info (assumes all frames in video have the same side)
    video_sides.append(vid_data["side"].iloc[0])

# Convert to NumPy array
X = np.array(video_sequences)
X_df = pd.DataFrame(X.reshape(-1, X.shape[-1]))  # Flatten sequences
X_df = X_df.interpolate(method="linear", axis=0).fillna(method="bfill")  # Linear fill, then backfill

X = X_df.to_numpy().reshape(X.shape)
# Normalize keypoint data
X_min = X.min(axis=(0, 1), keepdims=True)
X_max = X.max(axis=(0, 1), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)  # Avoid division by zero

print(f"Final dataset shape: {X.shape}")  # Expected: (num_videos, 40, num_features)

# Convert 'side' to numerical labels (0 = left, 1 = right)
side_labels = np.array([1 if side == "right" else 0 for side in video_sides])

# One-hot encode the labels
y = tf.keras.utils.to_categorical(side_labels, num_classes=2)
print(X)
# === Define LSTM Model using Functional API ===
input_layer = Input(shape=(NUM_FRAMES, X.shape[2]))

# LSTM layers
x = LSTM(64, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dropout(0.2)(x)

# Dense layers
x = Dense(16, activation="relu")(x)
output_layer = Dense(2, activation="softmax")(x)  # Predicts left or right

# Define model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train model (trial run)
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,shuffle=True)
