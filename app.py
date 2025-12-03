import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load the trained model (Kaggle working folder)
# -------------------------------
MODEL_PATH = "decision_tree_salary_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Make sure you ran the model training cell and saved the .pkl file.")
    st.stop()

# -------------------------------
# Load dataset (optional)
# -------------------------------
DATA_PATH = "/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("Salary Prediction using Decision Tree Regression")
st.write("Predict employee salary based on years of experience")

# User input
years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    # Decision Tree Regression does not require scaling
    predicted_salary = model.predict([[years_exp]])[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

# Optional: Show Dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

