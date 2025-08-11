import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/behavior_model.pkl')

# Load label encoders
label_encoders = joblib.load('models/label_encoders.pkl')

# Load expected training columns
feature_columns = joblib.load(open("models/feature_columns.pkl", "rb"))

# Test input (you can replace with dynamic values later)
input_data = {
    'A1': 'Yes',
    'A2': 'No',
    'A3': 'Yes',
    'A4': 'No',
    'A5': 'Yes',
    'A6': 'Yes',
    'A7': 'No',
    'A8': 'No',
    'A9': 'Yes',
    'A10_Autism_Spectrum_Quotient': 'No',
    'Sex': 'Male',
    'Ethnicity': 'White',
    'Age_Years': 5,
    'Jaundice': 'No',
    'Family_mem_with_ASD': 'Yes',
    'Speech Delay/Language Disorder': 'No',
    'Depression': 'No',
    'Anxiety_disorder': 'Yes',
    'Social/Behavioural Issues': 'Yes',
    'Genetic_Disorders': 'No'
}

# Fill missing features with default values (e.g. 'No' or 0)
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 'No' if col in label_encoders else 0  # use 0 for numeric if unknown

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Encode categorical features
for col, le in label_encoders.items():
    if col in df:
        df[col] = le.transform(df[col])

# Ensure correct column order
df = df[feature_columns]

# Predict
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]  # confidence score for "autistic"

# Output
result = {
    "prediction": "Autistic" if prediction == 1 else "Non-Autistic",
    "confidence": round(probability * 100, 2)
}

print(result)
