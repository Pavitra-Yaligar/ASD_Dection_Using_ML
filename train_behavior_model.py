import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
data = pd.read_csv("datasets/behavioral_data.csv", sep="\t", engine='python')

# Step 2: Rename columns for consistency
data.rename(columns={
    "A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4", "A5": "A5",
    "A6": "A6", "A7": "A7", "A8": "A8", "A9": "A9", 
    "A10_Autism_Spectrum_Quotient": "A10",
    "Age_Years": "age",
    "Sex": "gender",
    "Family_mem_with_ASD": "family_history",
    "Jaundice": "jaundice",
    "ASD_traits": "class_asd"
}, inplace=True)

# Step 3: Encode categorical columns
label_encoders = {}
categorical_columns = ['gender', 'jaundice', 'family_history', 'class_asd']
for col in categorical_columns:
    data[col] = data[col].astype(str).str.strip().str.lower()  # Normalize strings
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 4: Define features and target
features = [f"A{i}" for i in range(1, 11)] + ['gender', 'age', 'family_history', 'jaundice']
target = 'class_asd'

X = data[features]
y = data[target]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the model
with open("models/behavior_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Optional: Save label encoders
with open("models/behavior_label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

print("✅ Model and label encoders saved successfully.")
