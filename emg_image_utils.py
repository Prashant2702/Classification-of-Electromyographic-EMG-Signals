import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np

def create_emg_image(emg_data):
    """Create image from EMG data"""
    plt.figure(figsize=(6, 6))
    plt.plot(emg_data)
    plt.title("EMG Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return Image.open(img_buffer)
