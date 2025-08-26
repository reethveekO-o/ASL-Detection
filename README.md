# American Sign Language (ASL) Detector

This project uses a pretrained **TensorFlow** model (downloaded from [KaggleHub](https://www.kaggle.com/)) to recognize **American Sign Language (ASL)** hand signs in real time through a webcam feed.  

The application captures a region of interest (ROI) where the user places their hand, preprocesses the image, and runs inference to predict the corresponding ASL sign.

---

## 🚀 Features
- Real-time ASL sign recognition using webcam input  
- Pretrained deep learning model (TensorFlow 2)  
- 29 output classes: **A–Z, space, delete, nothing**  
- Visual guide (green box) to help users place their hand correctly  

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/reethveekO-o/ASL-Detection
   cd asl-detector
