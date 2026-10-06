import streamlit as st
import pandas as pd
import joblib

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

gender = st.selectbox(
    "Gender",
    options=["Female", "Male", "Other"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

hypertension = st.selectbox(
    "Hypertension",
    options=[0, 1],
    format_func=lambda value: "Yes" if value else "No"
)

heart_disease = st.selectbox(
    "Heart disease",
    options=[0, 1],
    format_func=lambda value: "Yes" if value else "No"
)

smoking_history = st.selectbox(
    "Smoking history",
    options=["No Info", "current", "ever", "former", "never", "not current"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=95.69,
    value=27.32
)

hb_a1c_level = st.number_input(
    "HbA1c level",
    min_value=3.5,
    max_value=9.0,
    value=5.8
)

blood_glucose_level = st.number_input(
    "Blood glucose level",
    min_value=80,
    max_value=300,
    value=140
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict"):

    # Create DataFrame with the same
    # feature names used during training
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hb_a1c_level],
        "blood_glucose_level": [blood_glucose_level]
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