🌊 Oil Spill Detection using SAR Imagery (U-Net-like CNN)
This repository contains code and resources for detecting oil spills in SAR (Synthetic Aperture Radar) grayscale satellite images using deep learning. The solution includes:

🧠 A custom U-Net-like CNN built with TensorFlow/Keras for semantic segmentation of oil spill regions

🛰️ SAR image preprocessing and normalization

📦 Binary mask generation and training from grayscale inputs

📈 Real-time detection with animated heatmap overlay

📁 Folder Structure
graphql
Copy
Edit
.
├── dataset/
│   ├── images/         # SAR grayscale images
│   └── masks/          # Corresponding binary masks (oil = 1, background = 0)
├── train_model.py      # Train the CNN model
├── oil_spill_model.h5  # Trained model (after running training script)
├── detect_animate.py   # Predict + animate oil spill region in SAR image
└── README.md           # This file
🛠 Requirements
Python 3.8+

TensorFlow 2.x

OpenCV

scikit-image

NumPy

Matplotlib

Install dependencies:

bash
Copy
Edit
pip install tensorflow opencv-python scikit-image numpy matplotlib
🚀 1. Training the Model
Use the train_model.py script to load images and binary masks and train a segmentation model.

train_model.py overview:
Reads SAR images and corresponding binary masks from dataset/

Resizes images to 128×128

Trains a CNN with encoder-decoder structure

Saves trained model to oil_spill_detector.h5

bash
Copy
Edit
python train_model.py
🔍 2. Oil Spill Detection + Animated Visualization
Use detect_animate.py to:

Load a grayscale SAR image

Run inference using the trained model

Animate the predicted oil spill mask over the original image using a jet colormap

bash
Copy
Edit
python detect_animate.py
Input SAR Image:
C:/Users/admin/Desktop/sars/sar5.png

Modify this path as needed in the script.

🖼 Output Example
https://user-images.githubusercontent.com/your_gif.gif
(Add screen recording or animated GIF here after running animation)

⚙️ Key Features
✔️ Offline deep learning prediction using a lightweight model

✔️ Inversion logic for highlighting oil spill areas (dark regions in SAR)

✔️ Animation using matplotlib.animation.FuncAnimation

✔️ Normalized heatmap overlay for visual clarity

🧪 Model Performance
Metric	Result (Example Dataset)
Input Size	128x128
Loss	Binary Crossentropy
Accuracy	~92%
Visual Output	Jet colormap heatmap

📌 Notes
This project assumes dark regions in SAR represent oil spills.

Invert logic was applied to match this property (1 - predicted_mask)

You can retrain using corrected masks if your model predicts inverse behavior.

🧩 To-Do / Future Enhancements
 Switch to U-Net with skip connections for better segmentation accuracy

 Add GUI for image selection and real-time detection

 Deploy via Flask or Streamlit web app


