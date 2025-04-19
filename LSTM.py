import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import matplotlib.pyplot as plt


# Paths to both CSV files
right_csv_path = "pose_data/right_goal/final_test/kicker_pose_keypoints.csv"
left_csv_path = "pose_data/left_goal/final_test/kicker_pose_keypoints.csv"
middle_csv_path ="pose_data/center_goal/final_test/kicker_pose_keypoints.csv"

# Load both datasets
df_right = pd.read_csv(right_csv_path)
df_left = pd.read_csv(left_csv_path)
df_middle=pd.read_csv(middle_csv_path)

# Modify the video_id to distinguish left and right sides
df_right["video_id"] = df_right["video_id"].astype(str) + "r"
df_left["video_id"] = df_left["video_id"].astype(str) + "l"
df_middle["video_id"] = df_middle["video_id"].astype(str) + "m"

# Assign a "side" column to differentiate data
df_right["side"] = "right"
df_left["side"] = "left"
df_middle["side"]= "middle"

# df_middle_dup = df_middle.copy()
# df_middle_dup["video_id"] = df_middle_dup["video_id"].astype(str) + "_dup"
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



#Class_Weighting
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(side_labels_raw),
#     y=side_labels_raw
# )
# class_weight_dict = dict(enumerate(class_weights))
# print("Class Weights:", class_weight_dict)
def plot_history(history, fold):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Fold {fold+1} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# === Cross-validation ===
def create_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    output_layer = Dense(3, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, side_labels_raw)):
    print(f"\nFold {fold + 1}")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_raw, y_val_raw = side_labels_raw[train_idx], side_labels_raw[val_idx]

    # === SMOTE inside CV fold ===
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    smote = SMOTE(sampling_strategy={0: sum(y_train_raw == 0) + 100}, random_state=42)
    X_train_resampled_flat, y_train_resampled_raw = smote.fit_resample(X_train_flat, y_train_raw)

    X_train_resampled = X_train_resampled_flat.reshape(-1, NUM_FRAMES, X.shape[2])
    y_train = tf.keras.utils.to_categorical(y_train_resampled_raw, num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=3)

    # === Downsampling the validation set to match the smallest class ===
    class_counts = np.bincount(y_val_raw)
    min_class_size = min(class_counts)

    # Create a new balanced validation set by undersampling the larger classes
    balanced_X_val = []
    balanced_y_val = []

    for class_label in np.unique(y_val_raw):
        # Get the indices for the current class
        class_indices = np.where(y_val_raw == class_label)[0]

        # Resample the current class to the size of the smallest class
        class_data = X_val[class_indices]
        class_labels = y_val_raw[class_indices]

        # Undersample to match the smallest class size
        class_data_resampled, class_labels_resampled = resample(
            class_data, class_labels,
            n_samples=min_class_size,  # Undersample to the size of the smallest class
            random_state=42
        )

        balanced_X_val.append(class_data_resampled)
        balanced_y_val.append(class_labels_resampled)

    # Concatenate all the resampled data into a single dataset
    balanced_X_val = np.vstack(balanced_X_val)
    balanced_y_val = np.concatenate(balanced_y_val)

    # Convert to one-hot encoding
    y_val_resampled = tf.keras.utils.to_categorical(balanced_y_val, num_classes=3)

    # === Model training ===
    model = create_model((NUM_FRAMES, X.shape[2]))
    # model.fit(X_train_resampled, y_train, epochs=50, batch_size=32, verbose=1)
    history = model.fit(
        X_train_resampled, y_train,
        epochs=50,
        batch_size=16,
        verbose=1,
        validation_data=(balanced_X_val, y_val_resampled)
    )

    preds = model.predict(balanced_X_val)
    preds_classes = np.argmax(preds, axis=1)
    val_true = np.argmax(y_val_resampled, axis=1)

    acc = accuracy_score(val_true, preds_classes)
    print(f"Validation Accuracy: {acc:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(val_true, preds_classes))

    print("Classification Report:")
    print(classification_report(val_true, preds_classes, target_names=["middle", "left", "right"]))

    accuracies.append(acc)



print(f"\nAverage Accuracy over {kfold.get_n_splits()} folds: {np.mean(accuracies):.4f}")
