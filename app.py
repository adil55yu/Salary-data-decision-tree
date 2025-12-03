import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load the trained model
# -------------------------------
with open("decision_tree_salary_model.pkl", "rb") as file:
    model = pickle.load(file)


# Initialize scaler (same as training)
scaler = StandardScaler()

# Load dataset to fit scaler (needed because DT was trained on scaled X)
df = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")
X = df[["YearsExperience"]]
scaler.fit(X)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Salary Prediction using Decision Tree Regression")
st.write("Predict employee salary based on years of experience")

# User input
years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    # Scale input
    years_exp_scaled = scaler.transform([[years_exp]])
    
    # Predict
    predicted_salary = model.predict(years_exp_scaled)[0]
    
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

# -------------------------------
# Optional: Show Dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)
