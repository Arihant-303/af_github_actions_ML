import streamlit as st
import numpy as np
import joblib

st.title("Diabetes Prediction App")

# Load the trained model
model = joblib.load('models/model.pkl')

# Create input fields for user to enter data
preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=0)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Create a button to make predictions
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.error("You are DIABETIC.")
    else:
        st.success("You are NOT DIABETIC.")