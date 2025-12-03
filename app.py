import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# File paths
# -------------------------------
MODEL_PATH = "/decision_tree_salary_model.pkl"
DATA_PATH = "/salary-data-simple-linear-regression/Salary_Data.csv"

# -------------------------------
# Load the trained model
# -------------------------------
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Make sure you have saved the .pkl file in Kaggle working folder.")
    st.stop()

# -------------------------------
# Load dataset (optional)
# -------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Salary Prediction using Decision Tree Regression")
st.write("Predict employee salary based on years of experience")

# User input
years_exp = st.number_input(
    "Enter Years of Experience:",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Predict button
if st.button("Predict Salary"):
    predicted_salary = model.predict([[years_exp]])[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

# Optional: Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

