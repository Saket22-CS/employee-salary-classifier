import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Load the trained model and preprocessing objects
model = joblib.load("best_model.pkl")
# Note: You should save these during training and load them here
# For now we'll recreate them (not ideal but works for demo)

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Work Class", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "NotListed"
])
education_num = st.sidebar.slider("Education Level (1-16)", 1, 16, 9)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband",
    "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Other"
])

# Preprocessing function
def preprocess_input(input_dict):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Label encode categorical features (must match training)
    cat_cols = ['workclass', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'native-country']

    for col in cat_cols:
        le = LabelEncoder()
        # Fit with possible categories (in real app, use saved encoders)
        le.fit(data[col])  # You should load your training data's encoders instead
        input_df[col] = le.transform(input_df[col])

    # Ensure all columns are in correct order
    expected_cols = ['age', 'workclass', 'educational-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'gender',
                    'hours-per-week', 'native-country']

    # Scale features (in real app, use saved scaler)
    scaler = MinMaxScaler()
    scaler.fit(data[expected_cols])  # Again, use your saved scaler
    input_scaled = scaler.transform(input_df[expected_cols])

    return input_scaled

# Build input dictionary
input_dict = {
    'age': age,
    'workclass': workclass,
    'educational-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

st.write("### 🔎 Input Data")
st.write(pd.DataFrame([input_dict]))

# Predict button
if st.button("Predict Salary Class"):
    try:
        processed_input = preprocess_input(input_dict)
        prediction = model.predict(processed_input)
        st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '<=50K'}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())

        # Preprocess batch data (same as single prediction)
        processed_batch = preprocess_input(batch_data)
        batch_preds = model.predict(processed_batch)

        batch_data['PredictedClass'] = ['>50K' if x == 1 else '<=50K' for x in batch_preds]
        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"❌ Batch processing error: {str(e)}")
