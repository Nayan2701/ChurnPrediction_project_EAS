# streamlit/app.py
import streamlit as st
import requests
import json
import os

st.title("👤 Customer Churn Prediction")

# Load Schema
with open("/app/data/data_schema.json") as f:
    schema = json.load(f)

user_input = {}

# Categorical Inputs
col1, col2 = st.columns(2)
with col1:
    user_input["Geography"] = st.selectbox("Geography", schema["categorical"]["Geography"]["unique_values"])
    user_input["Gender"] = st.selectbox("Gender", schema["categorical"]["Gender"]["unique_values"])
with col2:
    user_input["HasCrCard"] = st.selectbox("Has Credit Card?", [0, 1])
    user_input["IsActiveMember"] = st.selectbox("Is Active Member?", [0, 1])

# Numerical Inputs
for col, stats in schema["numerical"].items():
    user_input[col] = st.slider(col, stats["min"], stats["max"], stats["median"])

if st.button("Predict"):
    res = requests.post("http://api:8000/predict", json={"instances": [user_input]})
    if res.status_code == 200:
        pred = res.json()["predictions"][0]
        st.error("Churn Risk: HIGH") if pred == 1 else st.success("Churn Risk: LOW")
    else:
        st.error("API Error")