import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Prepare the dataset
def load_images_from_folder(folder):
    features = []
    labels = []

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)

            # Check if it's an image file
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read {img_path}")
                continue

            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features.append(img.flatten())
            labels.append(1 if label.lower() in ['asd', 'autistic'] else 0)

    return np.array(features), np.array(labels)


# Load dataset
X, y = load_images_from_folder('datasets/image_dataset/train')  # <-- Use your path here

# Step 2: Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Step 4: Save model and scaler
os.makedirs('models', exist_ok=True)
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('models/svm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 5: Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
