import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 pretrained (just for feature extraction)
print("[INFO] Loading YOLOv5 for feature extraction...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to(device)
yolo_model.eval()

# Load trained SVM model
print("[INFO] Loading SVM model...")
svm_model = joblib.load('models/svm_image_model.pkl')

# Feature extraction function
def extract_features_yolo(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = transforms.Resize((640, 640))(img)
        img = transforms.ToTensor()(img).unsqueeze(0)

        img = img.to(device)

        with torch.no_grad():
            # Use backbone for feature extraction
            backbone = yolo_model.model.model[:10]
            outputs = backbone(img)

        return outputs.view(-1).cpu().numpy()

    except Exception as e:
        print(f"[ERROR] Cannot extract features: {e}")
        return None

# Predict function
def predict_image(image_path):
    features = extract_features_yolo(image_path)
    if features is None:
        return "Error", 0.0

    prediction = svm_model.predict([features])[0]
    confidence = svm_model.predict_proba([features])[0][prediction]

    label = "Autistic" if prediction == 1 else "Non-Autistic"
    return label, round(confidence * 100, 2)

# Test (optional)
if __name__ == "__main__":
    test_image = "uploads/sample.jpg"
    label, confidence = predict_image(test_image)
    print(f"Prediction: {label} with confidence {confidence}%")
