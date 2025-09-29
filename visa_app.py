#Imports
import pandas as pd
import streamlit as st
import joblib
import os

# Load pipeline
model_path = "visa_pipeline.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("MODEL LOADED SUCCESSFULLY")
else:
    st.warning(f"Model file {model_path} does not exist")

# App Heading
st.title("US Visa Prediction")
st.write("Fill your details")
# Form Inputs
with st.form("USVisa_Form"):
    continent = st.selectbox("continent", ["Asia", "Africa","North America","Europe","South America","Oceania"])
    education_of_employee = st.selectbox("education_of_employee", ["High School", "Master's","Bachelor's","Doctorate"])
    has_job_experience = st.selectbox("has_job_experience", ["Y","N"])
    requires_job_training = st.selectbox("requires_job_training", ["Y","N"])
    no_of_employees = st.number_input("no_of_employees",min_value=0.0)
    yr_of_estab = st.number_input("yr_of_estab")
    region_of_employment = st.selectbox("region_of_employment", ["West","Northeast","South","Midwest","Island"])
    prevailing_wage = st.number_input("prevailing_wage",min_value=0)
    unit_of_wage = st.selectbox("unit_of_wage", ["Hour","Year","Week","Month"])
    full_time_position = st.selectbox("full_time_position", ["Y","N"])
    Submitted = st.form_submit_button("Submit")

# Create DataFrame
if Submitted:
    input_data = pd.DataFrame([{
        "continent": continent,
        "education_of_employee": education_of_employee,
        "has_job_experience": has_job_experience,
        "requires_job_training": requires_job_training ,
        "no_of_employees": no_of_employees,
        "yr_of_estab": yr_of_estab,
        "region_of_employment": region_of_employment,
        "prevailing_wage": prevailing_wage,
        "unit_of_wage": unit_of_wage,
        "full_time_position": full_time_position
     }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"US_Visa Probability : {probability*100:.2f}%")