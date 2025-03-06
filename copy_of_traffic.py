
# https://www.youtube.com/watch?v=ByED80IKdIU

import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


save_dir = "Full Data from LSTM"
os.makedirs(save_dir, exist_ok=True)

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")

# Open traffic video
video_path = "vids/testing.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize vehicle count storage
vehicle_counts = []
frame_times = []

frame_count = 0

# Define vehicle class IDs (COCO dataset)
vehicle_classes = [2, 3, 5, 7]  # Car, Truck, Bus, Motorcycle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO detection
    results = model(frame)

    # Count vehicles in the frame
    vehicle_count = sum(1 for obj in results[0].boxes.data if int(obj[-1]) in vehicle_classes)
    vehicle_counts.append(vehicle_count)

    # Calculate time in seconds
    frame_times.append(frame_count / fps)

    # Annotate frame with detected objects
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(1)  # Necessary to update the window

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
        break

cap.release()
cv2.destroyAllWindows()


# Save vehicle count data as CSV
df = pd.DataFrame({"Time (s)": frame_times, "Vehicle Count": vehicle_counts})
df.to_csv("vehicle_counts.csv", index=False)

print("✅ Vehicle count data saved as 'vehicle_counts.csv'!")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("vehicle_counts.csv")

# Plot vehicle count over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["Time (s)"], y=df["Vehicle Count"], marker='o', linestyle='-', color='b')

# Customize the plot
plt.title("Traffic Flow Over Time", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Number of Vehicles", fontsize=12)
plt.grid(True)
plt.show()

save_path = os.path.join(save_dir, "Traffic Flow Over Time.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high resolution

# Summary statistics
mean_vehicles = df["Vehicle Count"].mean()
max_vehicles = df["Vehicle Count"].max()
min_vehicles = df["Vehicle Count"].min()

print(f"✅ Average Vehicle Count: {mean_vehicles:.2f}")
print(f"🚗 Peak Traffic (Max Vehicles): {max_vehicles}")
print(f"🟢 Least Traffic (Min Vehicles): {min_vehicles}")

# Find peak congestion times
peak_times = df[df["Vehicle Count"] == max_vehicles]["Time (s)"].values
print(f"⚠️ Peak Traffic Time(s): {peak_times}")

# Define congestion threshold (e.g., > 80% of max vehicles)
congestion_threshold = max_vehicles * 0.8
congested_times = df[df["Vehicle Count"] >= congestion_threshold]

if not congested_times.empty:
    print("🚦 High Congestion Detected at:")
    print(congested_times)
else:
    print("✅ No significant congestion detected.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load vehicle count data
df = pd.read_csv("vehicle_counts.csv")

# Normalize vehicle count (LSTMs work better with scaled data)
scaler = MinMaxScaler()
df["Vehicle Count"] = scaler.fit_transform(df[["Vehicle Count"]])

# Convert time column to an index (LSTM needs sequential data)
df.set_index("Time (s)", inplace=True)

# Plot the preprocessed data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Vehicle Count"], label="Normalized Vehicle Count", color='b')
plt.title("Normalized Traffic Data for LSTM")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Vehicle Count")
plt.legend()
plt.grid(True)
plt.show()

print("✅ Data Preprocessed for LSTM!")

save_path = os.path.join(save_dir, "Normalized Traffic Data for LSTM.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high resolution

def create_sequences(data, seq_length=30):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # Last 'seq_length' values
        Y.append(data[i + seq_length])  # Next value
    return np.array(X), np.array(Y)

# Define sequence length (e.g., last 30 seconds)
SEQ_LENGTH = 30

# Create sequences for LSTM
X, Y = create_sequences(df["Vehicle Count"].values, SEQ_LENGTH)

# Reshape X for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"✅ Data Shape for LSTM -> X: {X.shape}, Y: {Y.shape}")

from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

print(f"✅ Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# !pip install tensorflow numpy pandas scikit-learn matplotlib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization

# Define the improved LSTM model
model = Sequential([
    
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1))),  # Bidirectional LSTM
    BatchNormalization(),  # Normalize activations
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),  # Extra dense layer
    Dense(1)  # Output layer
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])

# Model summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot training loss
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Fine-Tuned LSTM Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

save_path = os.path.join(save_dir, "Fine-Tuned LSTM Model Training Loss.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high resolution

model.save

model.save('traffic_prediction_model.keras')

# Make predictions
predictions = model.predict(X_test)

# Reverse normalization (convert back to original scale)
predictions = scaler.inverse_transform(predictions)
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(Y_test_actual, label="Actual Vehicle Count", color="blue")
plt.plot(predictions, label="Predicted Vehicle Count", color="red", linestyle="dashed")
plt.title("Fine-Tuned LSTM Traffic Prediction")
plt.xlabel("Time")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)
plt.show()

save_path = os.path.join(save_dir, "Fine-Tuned LSTM Traffic Prediction.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high resolution