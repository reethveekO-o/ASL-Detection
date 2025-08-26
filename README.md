
# 🖐️ American Sign Language (ASL) Detector

## **Overview**

This project performs **real-time American Sign Language (ASL) recognition** using a pretrained deep learning model.  
It takes webcam input, extracts the region of interest (ROI) containing the hand, preprocesses it, and predicts the corresponding ASL sign.  

The system is designed as a simple, interactive tool to demonstrate **computer vision + deep learning** for sign language recognition.

---

## **Features**

- 📷 **Webcam Integration** → Real-time sign capture  
- 🤖 **Pretrained TensorFlow Model** (from KaggleHub)  
- 🔤 **29 Supported Classes**: A–Z, space, delete, nothing  
- 🟩 **ROI Guide** → On-screen bounding box for proper hand placement  

---

## **Installation & Usage**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/asl-detector
   cd asl-detector
   ```


2. **Check Python version**
   Ensure you are using **Python >=3.9 and <3.12**.

   ```bash
   python --version
   ```

3. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the ASL detector**

   ```bash
   python asl_detector.py
   ```

---

## **Controls**

* **SPACE** → Capture hand sign and classify
* **ESC** → Exit the program

⚠️ Make sure your hand is inside the **green rectangle** for accurate detection.

---

## **Architecture**

The system follows a simple pipeline:

1. **Input** → Webcam stream
2. **ROI Selection** → Green rectangle guides user to place their hand
3. **Preprocessing** → Resize to (224×224), normalize, RGB conversion
4. **Prediction** → TensorFlow pretrained model outputs probabilities across 29 classes
5. **Result** → Predicted ASL sign is displayed in console

---

## **Requirements**

* Python 3.9 – 3.11
* Webcam

Dependencies are listed in **requirements.txt**.

---

## **Acknowledgements**

* Pretrained ASL model by [sayannath235](https://www.kaggle.com/sayannath235) on Kaggle
* [KaggleHub](https://github.com/Kaggle/kagglehub) for model downloading
* TensorFlow & OpenCV for real-time recognition

---
