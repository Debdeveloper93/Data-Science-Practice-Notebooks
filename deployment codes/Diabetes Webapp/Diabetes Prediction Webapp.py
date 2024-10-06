# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:59:29 2024

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("C:/Users/delve/OneDrive/Desktop/Projects/trained_model.sav", "rb"))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # giving a title
    st.title("Diabetes Prediction Webapp")

    # getting the input data from the user
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Level of Skin Thickness")
    Insulin = st.text_input("Level of Insulin")
    BMI = st.text_input("Value of BMI")
    DiabetesPedigreeFunction = st.text_input("Value of Diabetes Pedigree Function")
    Age = st.text_input("Age of the person")

    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
