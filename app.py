import streamlit as st
import pandas as pd
import pickle

# Kaggle working folder path
MODEL_PATH = "/kaggle/working/decision_tree_salary_model.pkl"
DATA_PATH = "/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv"

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Streamlit app
st.title("Salary Prediction using Decision Tree Regression")
st.write("Predict employee salary based on years of experience")

years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    predicted_salary = model.predict([[years_exp]])[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

if st.checkbox("Show Dataset"):
    st.dataframe(df)
