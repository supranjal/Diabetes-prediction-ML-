import streamlit as st
import pandas as pd
import joblib

# Import custom preprocessing classes
# Required for loading the saved pipeline
import src.preprocessing


# -----------------------------------
# Load trained pipeline
# -----------------------------------

loaded_pipeline = joblib.load(
    "models/diabetes_prediction_pipeline.pkl"
)


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)


# -----------------------------------
# App title
# -----------------------------------

st.title("Diabetes Prediction System")

st.write(
    "Enter the patient's information below to predict "
    "the likelihood of diabetes."
)


# -----------------------------------
# Input fields
# -----------------------------------

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict"):

    # Create DataFrame with the same
    # feature names used during training
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Make prediction using the saved pipeline
    prediction = loaded_pipeline.predict(input_data)

    # Get probability of diabetes
    probability = loaded_pipeline.predict_proba(input_data)[0][1]

    # Display prediction
    if prediction[0] == 1:
        st.error(
            "The model predicts that the patient is likely to have diabetes."
        )
    else:
        st.success(
            "The model predicts that the patient is unlikely to have diabetes."
        )

    # Display probability
    st.write(
        f"Probability of diabetes: **{probability * 100:.2f}%**"
    )