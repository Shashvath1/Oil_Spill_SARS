import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model
from skimage import io, exposure

# === Load the trained model ===
model = load_model("oil_spill_model.h5")  # ← Update your model path

# === Load grayscale SAR image ===
image_path = r"C:\Users\admin\Desktop\sars\sar5.png"
gray_image = io.imread(image_path, as_gray=True)
gray_image = exposure.equalize_adapthist(gray_image)

# === Resize and prepare input ===
input_size = (128, 128)  # Change if needed
resized = cv2.resize(gray_image, input_size)
input_img = np.expand_dims(resized, axis=(0, -1))  # Shape: (1, H, W, 1)

# === Predict oil spill mask ===
pred_mask = model.predict(input_img)[0, :, :, 0]

# === Invert the mask: higher value → more likely oil spill (dark)
pred_mask_inverted = 1.0 - pred_mask

# Normalize to [-0.1, 0.1] for colormap
pred_scaled = np.interp(pred_mask_inverted, (pred_mask_inverted.min(), pred_mask_inverted.max()), [-0.1, 0.1])

# Resize to match original image size
spill_map = cv2.resize(pred_scaled, (gray_image.shape[1], gray_image.shape[0]))

# === Set up animation ===
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap("jet")
img = ax.imshow(spill_map * 0, cmap=cmap, vmin=-0.1, vmax=0.1)
ax.axis("off")
ax.set_title("Oil Spill Detection Animation")

# === Animate the overlay ===
n_frames = 30
def update(frame):
    factor = (frame + 1) / n_frames
    animated_map = spill_map * factor
    img.set_data(animated_map)
    ax.set_title(f"Oil Spill Detection Animation\nFrame {frame+1}/{n_frames}")
    return [img]

ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
plt.tight_layout()
plt.show()
