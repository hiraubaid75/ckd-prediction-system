🩺 Chronic Kidney Disease (CKD) Prediction System

🚀 Project Overview

Chronic Kidney Disease (CKD) Prediction System is a machine learning-powered web application built with Python, Scikit-learn, and Streamlit.
It predicts the likelihood of CKD based on patient medical attributes, providing a fast and interactive interface for healthcare professionals.

🔍 Features

✅ Predict CKD using Random Forest Classifier

✅ Interactive and user-friendly Streamlit interface

✅ Handles missing values and performs basic preprocessing

✅ Displays prediction as CKD / Not CKD

✅ Easy to deploy as a web app

📊 Dataset

Source: https://drive.google.com/file/d/1e1S2rNn0_Bjv3XrUthw_M7MFoGuYcQE1/view

Rows: 30000

Columns: 24 medical attributes (numeric & categorical)
Numeric Columns
Continuous Numeric (float64 / int64 but measured values)
•	age
•	blood_pressure
•	specific_gravity
•	blood_glucose_random
•	blood_urea
•	serum_creatinine
•	sodium
•	potassium
•	hemoglobin
•	packed_cell_volume
•	white_blood_cell_count
•	red_blood_cell_count
Discrete Numeric (int64 but categorical-like — ordered levels)
•	albumin (protein level in urine, usually 0–5 scale)
•	sugar (sugar level in urine, usually 0–5 scale)
________________________________________
Target Column
•	ckd (0 = no CKD, 1 = CKD) → numeric but categorical (binary classification).
________________________________________
✅ So in short:
•	Categorical features → 10 columns
•	Numeric features → 12 columns (+2 ordinal-like: albumin, sugar)
•	Target → ckd

Purpose: Used for training the Random Forest model to predict CKD

💻 Installation

Clone the repository

git clone https://github.com/hiraubaid75/ckd-prediction-system.git
cd ckd-prediction-system


Create a virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

▶️ Usage

Launch the Streamlit app:

streamlit run Streamlit_app.py


Enter patient medical information in the input fields.

Click Predict to see the result.

🧠 Model Details

Algorithm: Random Forest Classifier

Saved Model: best_random_forest_model.pkl

Libraries Used: scikit-learn, pandas, numpy, streamlit

📁 Folder Structure
ckd-prediction-system/
│
├── Streamlit_app.py                       # Streamlit application
├── best_random_forest_model.pkl  # Trained ML model
├── Chronic_Kidney_disease_dataset.csv # Dataset
├── requirements.txt             # Python dependencies
├── images folder               # App screenshots (optional)
└── README.md                    # Project documentation

🎯 Future Enhancements

Add more ML algorithms for performance comparison

Integrate data visualizations for better insights

Deploy on cloud platforms (Streamlit Cloud)

Implement user authentication for secure access

🖼 Visual Preview

see the images folder for the 

👤 Author

Hira Barlas


https://github.com/hiraubaid75
