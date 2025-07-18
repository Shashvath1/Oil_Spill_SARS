# 🌊 Oil Spill Detection in SAR Imagery using Deep Learning

This repository provides a complete pipeline to **train a CNN-based segmentation model** and **visualize detected oil spills** in SAR (Synthetic Aperture Radar) grayscale images with **animated heatmap overlays**.

The system includes:
- 🧠 A lightweight encoder-decoder CNN for segmentation
- 🌌 Detection of oil spill regions (typically dark in SAR images)
- 📦 Offline prediction and animation using matplotlib
- 🛰️ Adaptable for real-time SAR image analysis

---

## 📁 Folder Structure

├── dataset/
│ ├── images/ # SAR grayscale images
│ └── masks/ # Binary masks (oil spill = white, background = black)
├── train_model.py # Training script for CNN model
├── detect_animate.py # Detection + animated visualization script
├── oil_spill_detector.h5 # Trained Keras model (saved after training)
└── README.md


---

## 🔧 Requirements

Install dependencies with pip:

```bash
pip install tensorflow numpy opencv-python scikit-image matplotlib


Run the train_model.py script to train your oil spill segmentation model:

python train_model.py


📌 This will:

Load SAR images and masks from dataset/images and dataset/masks

Normalize and resize them to 128×128

Train a U-Net-like CNN

Save the trained model to oil_spill_detector.h5

🛰️ Running Detection + Heatmap Animation
Use detect_animate.py to detect oil spills in a new SAR grayscale image and animate the prediction heatmap:

bash
Copy
Edit
python detect_animate.py
💡 What it does:

Loads a test SAR image (modify image_path in the script)

Preprocesses and feeds it to the trained model

Inverts the predicted mask (dark = oil)

Creates a heatmap overlay animated over time

🎞 Sample Output:
Add a GIF or screenshot here (e.g. assets/animation.gif)

🧠 Model Architecture
text
Copy
Edit
Input (128x128x1)
↓
Conv2D → MaxPooling
↓
Conv2D → MaxPooling
↓
Conv2D → UpSampling
↓
Conv2D → UpSampling
↓
Conv2D (1x1, Sigmoid) → Output Mask (128x128)
Loss: binary_crossentropy
Optimizer: Adam
Final Activation: Sigmoid

🔬 Sample Prediction Flow
Load SAR image and trained model

Predict binary mask for oil spill region

Invert the output (since oil = dark)

Animate prediction using a jet colormap

📌 Notes
Make sure binary masks match the filenames of input images.

Model assumes dark patches in SAR images represent oil spill regions.

You can retrain with flipped labels if prediction appears reversed.

📈 Performance Overview
Feature	Result
Input size	128 × 128
Training Epochs	20
Accuracy (Validation)	~92%
Output Format	Binary mask (oil = 1)
Overlay Visualization	Animated jet heatmap

🧩 Future Enhancements
 Switch to full U-Net with skip connections

 Add CLI arguments for batch prediction

 GUI/Web-based visualization with Streamlit

 Use real-world SAR datasets from ESA/NASA


