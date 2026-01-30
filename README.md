# ASD_Dection_Using_ML

🧠 Autism Spectrum Disorder (ASD) Detection System
📌 Project Overview

This project is an end-to-end Autism Spectrum Disorder (ASD) Detection System that combines behavioral data analysis and image-based classification to assist in early screening of ASD.
The system is built as a web application with an intuitive user interface and machine learning–based prediction models.

🎯 Objectives

Early identification of ASD using behavioral and visual indicators
Provide confidence scores and visual insights for predictions
Support behavioral, image-based, and combined ASD detection modes
Deliver an easy-to-use platform for non-technical users

🛠️ Technologies Used

Python
Flask (Web Framework)
Power BI / Chart.js (Visualization)
Machine Learning: Random Forest
Deep Learning: YOLOv5 (Image Classification)
HTML, CSS, Bootstrap
SQLite3

🧠 Machine Learning Models
1️⃣ Behavioral Classification
Model: Random Forest Classifier

Inputs: 
  10 behavioral questions
  Age
  Gender
  Family history
  Jaundice

Output:
ASD probability (%)
Severity level (Low / Moderate / High)

2️⃣ Image-Based Classification
Model: YOLOv5

Input:
Child facial image

Output:
ASD / Non-ASD classification
Bounding box visualization
Confidence score

📂 Project Structure
ASD_Detection_Project/
│
├── app.py
├── predict_yolo.py
├── train_behavior_model.py
├── train_random_forest.py
│
├── datasets/
│   ├── behavioral_data.csv
│   ├── train/
│   ├── test/
│   └── valid/
│
├── models/
│   ├── behavior_model.pkl
│   ├── rf_model.pkl
│   └── best.pt
│
├── static/
│   ├── css/
│   ├── images/
│   └── videos/
│
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── questions.html
│   ├── image_upload.html
│   ├── result.html
│   └── dashboard.html
│
└── README.md

📊 Features

Secure login & signup system
Behavioral questionnaire (10 questions)
Image upload for facial analysis
Interactive confidence charts (Chart.js)
Severity level indication
Downloadable PDF report
ASD awareness and prevention video
Modern Bootstrap-based UI

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/ASD-Detection-System.git
cd ASD-Detection-System

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

📈 Results & Visualization

Behavioral confidence shown using bar/pie charts

Image prediction confidence with bounding box overlay

Combined ASD probability score

Exportable PDF summary report
