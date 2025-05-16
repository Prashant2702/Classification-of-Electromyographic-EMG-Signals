import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load dataset
file_path = "Book1.csv"  # Replace with your dataset file
df = pd.read_csv(file_path)

# Extract features (EMG channels) and labels
time_series_columns = df.columns[1:-2]  # Ignore 'time', 'class', and 'label'
labels = df["label"].values  # Assuming 'label' is the target column
num_features = len(time_series_columns)  # Number of columns

# Determine the best reshape size (square-like or closest possible)
reshape_size = int(np.sqrt(num_features))
if reshape_size * reshape_size != num_features:
    reshape_size = num_features  # Keep it 1D if not a perfect square

# Create folders for saving images
output_dir = "EMG_Images"
class_0_dir = os.path.join(output_dir, "Class_0")
class_1_dir = os.path.join(output_dir, "Class_1")

os.makedirs(class_0_dir, exist_ok=True)
os.makedirs(class_1_dir, exist_ok=True)

# Convert EMG signals to images
for i, row in tqdm(enumerate(df[time_series_columns].values), total=len(df)):
    label = labels[i]
    save_dir = class_0_dir if label == 0 else class_1_dir  # Save based on class

    # Normalize signal (0-255 grayscale)
    img_data = np.array(row, dtype=np.float32)
    img_data = ((img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255).astype(np.uint8)

    # Reshape dynamically
    if reshape_size * reshape_size == num_features:
        img_data = img_data.reshape(reshape_size, reshape_size)  # Square shape
    else:
        img_data = img_data.reshape(1, num_features)  # 1D image if not square

    # Save as image
    img_path = os.path.join(save_dir, f"emg_{i}.png")  # Save as PNG format
    plt.imsave(img_path, img_data, cmap="gray", format="png")

print(f"✅ Images saved in '{class_0_dir}' and '{class_1_dir}'.")
