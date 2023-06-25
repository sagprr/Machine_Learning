#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:51:27 2023

@author: sagarpanwar
"""

import numpy as np 
import pickle 
import streamlit as st



import pickle

with open('/Users/sagarpanwar/Desktop/Downloads/trained_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)



# creating a function for Prediction

def diabetes_prediction(input_data):
    
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      print('The person is not diabetic')
      
    else:
      print('The person is diabetic')
      
      
    
def main():
    
    #giving a title
    st. title( 'Diabetes Prediction Web App' )
    
              
    # getting the input data from the user
      
    Pregnancies = st.text_input('Number of Pregrnancies')
    Glucose = st.text_input('Number of Glucose')
    BloodPressure = st.text_input('Number of BloodPressure')
    SkinThickness = st.text_input('Number of SkinThickness')
    Insulin = st.text_input('Number of Insulin')
    BMI = st.text_input('Number of BMI')
    DiabetesPedigreeFunction = st.text_input('Number of DiabetesPedigreeFunction')
    Age = st.text_input('Number of Age')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Button'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    

if __name__ == '__main__' :
    main()
    
    
      
     
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      