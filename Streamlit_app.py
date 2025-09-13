# app.py
import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="wide")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    with open("best_random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Dashboard")
page = st.sidebar.radio("", ["🏠 Home", "🔍 Prediction", "📊 Visualizations", "👤 About Me"])

# ----------------------------
# Home Page
# ----------------------------
if page == "🏠 Home":
    st.title("🩺 Chronic Kidney Disease Prediction App")
    st.subheader("Built by Hira Barlas")
    st.write("""
    This project predicts the likelihood of **Chronic Kidney Disease (CKD)** 
    using patient clinical data and a Random Forest model.

    ✅ Features:
    - Prediction based on **24 clinical parameters**
    - Interactive **visualizations** of model performance
    - Beginner-friendly explanations

    Use the sidebar to navigate 🚀
    """)

# ----------------------------
# Prediction Page
# ----------------------------
elif page == "🔍 Prediction":
    st.title("🔍 CKD Prediction")
    st.write("Fill in patient details below and click **Predict CKD**.")

    # Two-column layout
    col1, col2 = st.columns(2)

    # ----------------------------
    # Left column inputs
    # ----------------------------
    with col1:
        age = st.number_input("Age", 0, 120, 50, step=1, key="age")
        gender = st.selectbox("Gender", ["male", "female"], key="gender")
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 80, step=1, key="blood_pressure")
        specific_gravity = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025], key="specific_gravity")
        albumin = st.selectbox("Albumin", [0,1,2,3,4,5], key="albumin")
        sugar = st.selectbox("Sugar", [0,1,2,3,4,5], key="sugar")
        pus_cell = st.selectbox("Pus Cell", ["normal", "abnormal"], key="pus_cell")
        pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], key="pus_cell_clumps")
        bacteria = st.selectbox("Bacteria", ["present", "notpresent"], key="bacteria")
        blood_glucose_random = st.number_input("Blood Glucose Random", 0.0, 500.0, 120.0, step=1.0, key="blood_glucose_random")
        blood_urea = st.number_input("Blood Urea", 0.0, 500.0, 40.0, step=1.0, key="blood_urea")
        serum_creatinine = st.number_input("Serum Creatinine", 0.0, 50.0, 1.2, step=0.01, key="serum_creatinine")

    # ----------------------------
    # Right column inputs
    # ----------------------------
    with col2:
        sodium = st.number_input("Sodium", 0.0, 200.0, 135.0, step=1.0, key="sodium")
        potassium = st.number_input("Potassium", 0.0, 20.0, 4.5, step=0.01, key="potassium")
        hemoglobin = st.number_input("Hemoglobin", 0.0, 30.0, 15.0, step=0.1, key="hemoglobin")
        packed_cell_volume = st.number_input("Packed Cell Volume", 0, 70, 40, step=1, key="packed_cell_volume")
        white_blood_cell_count = st.number_input("White Blood Cell Count", 0, 50000, 8000, step=1, key="white_blood_cell_count")
        red_blood_cell_count = st.number_input("Red Blood Cell Count", 0.0, 10.0, 4.5, step=0.01, key="red_blood_cell_count")
        hypertension = st.selectbox("Hypertension", ["yes", "no"], key="hypertension")
        diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["yes", "no"], key="diabetes_mellitus")
        coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["yes", "no"], key="coronary_artery_disease")
        appetite = st.selectbox("Appetite", ["good", "poor"], key="appetite")
        anemia = st.selectbox("Anemia", ["yes", "no"], key="anemia")
        pedal_edema = st.selectbox("Pedal Edema", ["yes", "no"], key="pedal_edema")
        red_blood_cells = st.selectbox("Red Blood Cells", ["normal", "abnormal"], key="red_blood_cells")

    # ----------------------------
    # Prepare DataFrame with exact column names
    # ----------------------------
    features_dict = {
        "age": [age],
        "blood_pressure": [blood_pressure],
        "specific_gravity": [specific_gravity],
        "albumin": [albumin],
        "sugar": [sugar],
        "blood_glucose_random": [blood_glucose_random],
        "blood_urea": [blood_urea],
        "serum_creatinine": [serum_creatinine],
        "sodium": [sodium],
        "potassium": [potassium],
        "hemoglobin": [hemoglobin],
        "packed_cell_volume": [packed_cell_volume],
        "white_blood_cell_count": [white_blood_cell_count],
        "red_blood_cell_count": [red_blood_cell_count],
        "red_blood_cells": [red_blood_cells],
        "pus_cell": [pus_cell],
        "pus_cell_clumps": [pus_cell_clumps],
        "bacteria": [bacteria],
        "hypertension": [hypertension],
        "diabetes_mellitus": [diabetes_mellitus],
        "coronary_artery_disease": [coronary_artery_disease],
        "appetite": [appetite],
        "pedal_edema": [pedal_edema],
        "anemia": [anemia],
        "gender": [gender]
    }

    features_df = pd.DataFrame(features_dict)

    # ----------------------------
    # Predict button
    # ----------------------------
    if st.button("Predict CKD"):
        try:
            proba = model.predict_proba(features_df)[0][1]
            st.success(f"Probability of CKD: {proba:.2f}")
            if proba >= 0.5:
                st.warning("The patient may have CKD. Please consult a doctor.")
            else:
                st.info("The patient is unlikely to have CKD.")
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------------
# Visualizations Page
# ----------------------------
elif page == "📊 Visualizations":
    st.title("📊 Model Performance Visuals")

    images = {
        "Confusion Matrix": "images/confusion_matrix_rf.png",
        "ROC Curve": "images/roc_curve_rf.png",
        "Feature Importance": "images/feature_importance_rf.png",
        "Probability Distribution": "images/probability_distribution_rf.png"
    }

    for caption, path in images.items():
        # Big, bold centered title
        st.markdown(f"<h2 style='text-align:center; color: black;'>{caption}</h2>", unsafe_allow_html=True)
        try:
            # Set a fixed width for better clarity
            st.image(path, width=700)
        except:
            st.info(f"{caption} image not found.")
        st.markdown("---")

# ----------------------------
# About Me Page
# ----------------------------
elif page == "👤 About Me":
    st.title("👤 About Me")
    st.write("""
    **Name:** Hira Barlas  
    **Location:** UAE  
    **Current Role:** Data Analyst Trainee, attending a Data Analytics Bootcamp  
    **Project:** Chronic Kidney Disease Prediction  
    **GitHub:** [View Project](https://github.com/hiraubaid75/ckd-prediction-system)  
    **Contact:** hiraubaid95@gmail.com | [LinkedIn](https://www.linkedin.com/in/hira-barlas/)
    """)
    st.image("https://avatars.githubusercontent.com/u/104772634?v=4", width=200)



