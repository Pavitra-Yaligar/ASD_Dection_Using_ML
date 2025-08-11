import torch
import cv2
import numpy as np
import os
import pickle

# ✅ Correct YOLOv5 local imports
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# === Step 1: Load YOLOv5 model ===
device = select_device('')
yolo_model = DetectMultiBackend('best.pt', device=device)
yolo_model.eval()

# === Step 2: Load SVM model ===
with open('models/svm_image_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# === Step 3: Predict function ===
def predict_image(image_path):
    # Load and preprocess image
    img0 = cv2.imread(image_path)
    assert img0 is not None, f"Image not found: {image_path}"

    # Resize for YOLOv5
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalize to 0-1
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Run YOLOv5 model
    pred = yolo_model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    features = []
    for det in pred:
        if len(det):
            # Rescale boxes to original image size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                face = img0[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (64, 64)).flatten()
                features.append(face)

    if not features:
        return "No face detected", None

    # Predict using SVM
    features_np = np.array(features)
    predictions = svm_model.predict(features_np)
    prediction = max(set(predictions), key=list(predictions).count)

    return prediction, pred

# === Step 4: For testing only ===
if __name__ == '__main__':
    test_path = 'uploads/sample.jpg'  # 🟡 Replace with actual image
    result, _ = predict_image(test_path)
    print("Predicted Class:", result)
