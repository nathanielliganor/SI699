import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your real preprocessed data here
base_path = os.path.dirname(os.path.abspath(__file__))

df_right = pd.read_csv(os.path.join(base_path, "right.csv"))
df_left = pd.read_csv(os.path.join(base_path, "left.csv"))
df_middle = pd.read_csv(os.path.join(base_path, "center.csv"))

# Modify the video_id to distinguish left and right sides
df_right["video_id"] = df_right["video_id"].astype(str) + "r"
df_left["video_id"] = df_left["video_id"].astype(str) + "l"
df_middle["video_id"] = df_middle["video_id"].astype(str) + "m"

# Assign a "side" column to differentiate data
df_right["side"] = "right"
df_left["side"] = "left"
df_middle["side"]= "middle"

# Combine both CSVs
df = pd.concat([df_right, df_left,df_middle], ignore_index=True)
df = df.sort_values(by=["video_id", "frame"]).reset_index(drop=True)
print(df.groupby("side").video_id.nunique())
print(f"Total videos found: {df['video_id'].nunique()}")

NUM_FRAMES = 30
video_ids = df["video_id"].unique()
keypoint_cols = [col for col in df.columns if "kp_" in col]
zero_percentage = (df[keypoint_cols] == 0).mean()
columns_to_use = zero_percentage[zero_percentage <= 0.50].index.tolist()
keypoint_cols = [col for col in columns_to_use]

video_sequences = []
video_sides = []

for vid in video_ids:
    vid_data = df[df["video_id"] == vid]
    vid_data = vid_data.iloc[-NUM_FRAMES:] if len(vid_data) >= NUM_FRAMES else vid_data
    while len(vid_data) < NUM_FRAMES:
        vid_data = pd.concat([vid_data, vid_data.iloc[-1:]])
    video_sequences.append(vid_data[keypoint_cols].values)
    video_sides.append(vid_data["side"].iloc[0])

X = np.array(video_sequences)
X_df = pd.DataFrame(X.reshape(-1, X.shape[-1]))
X_df = X_df.interpolate(method="linear", axis=0).fillna(method="bfill")
X = X_df.to_numpy().reshape(X.shape)

#Normalize
X_min = X.min(axis=(0, 1), keepdims=True)
X_max = X.max(axis=(0, 1), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)

print(f"Final dataset shape: {X.shape}")

# Labels: raw and one-hot
side_labels_raw = np.array([
    2 if side == "right" else 1 if side == "left" else 0 if side == "middle" else -1
    for side in video_sides
])
print(pd.Series(side_labels_raw).value_counts())
# For demonstration, I'm generating random data
np.random.seed(42)
num_samples_per_class = 50
keypoint_cols = [f"kp_{i}_{axis}" for i in range(10, 20) for axis in ['x', 'y']]  # 20 features
X = np.random.rand(num_samples_per_class * 3, len(keypoint_cols))
y = np.array([0]*num_samples_per_class + [1]*num_samples_per_class + [2]*num_samples_per_class)  # middle, left, right

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train a Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=["middle", "left", "right"])

# Show results
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=keypoint_cols,
    class_names=["middle", "left", "right"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Predicting Kick Direction from Keypoints")
plt.savefig("my_plot.png") 
plt.show()


import os
print(os.getcwd())