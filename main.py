import cv2
import numpy as np
import tensorflow as tf
import kagglehub

# Load the pretrained ASL model
model_path = kagglehub.model_download("sayannath235/american-sign-language/tensorFlow2/american-sign-language")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# ASL class labels
class_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "space", "delete", "nothing"
]

def preprocess_image(frame, img_size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, img_size)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Webcam initialized. Press SPACE to capture and classify hand sign, or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rect_width = w // 4
    rect_height = h // 3
    x1, y1 = 20, h // 3
    x2, y2 = x1 + rect_width, y1 + rect_height

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place Hand Inside", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("ASL Detector", frame)
    key = cv2.waitKey(1)
    
    if key == 32:  # SPACE key
        hand_region = frame[y1:y2, x1:x2]
        input_image = preprocess_image(hand_region)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)
        predicted_sign = class_labels[predicted_class]
        print(f"Predicted ASL Sign: {predicted_sign}")

    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
