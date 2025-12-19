import streamlit as st
import requests
import json
import os

API_URL = "https://churnprediction-project-eas.onrender.com/predict"
SCHEMA_PATH = "/app/data/data_schema.json"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Predictor")

options = {}
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, 'r') as f:
        options = json.load(f)

with st.form("churn_form"):
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    geography = st.selectbox("Geography", options.get("categorical", {}).get("Geography", {}).get("unique_values", ["France", "Spain", "Germany"]))
    gender = st.selectbox("Gender", options.get("categorical", {}).get("Gender", {}).get("unique_values", ["Male", "Female"]))
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, value=0.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_card = st.selectbox("Has CrCard?", [0, 1])
    is_active = st.selectbox("Is Active?", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, value=50000.0)
    
    if st.form_submit_button("Predict"):
        payload = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": has_card, "IsActiveMember": is_active, "EstimatedSalary": salary
        }
        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                prob = res.json()["churn_probability"]
                st.success(f"Churn Probability: {prob:.1%}")
            else:
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                  prob = res.json()["churn_probability"]
                  st.success(f"Churn Probability: {prob:.1%}")
                else:
            # THIS IS THE NEW PART
                 st.error(f"Error {res.status_code}")
                 st.write(res.text)
        except Exception as e:
            st.error(f"Connection Error: {e}")
