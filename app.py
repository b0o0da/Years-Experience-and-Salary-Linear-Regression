import streamlit as st
import pickle
import numpy as np
import os
import subprocess
import sys
from sklearn.metrics import mean_absolute_error
import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv("Salary.csv")
X = df["YearsExperience"]
y = df["Salary"]
X=pd.DataFrame(X)
y=pd.DataFrame(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=.2 , random_state=101 )


model_path = "model.pkl"  
with open(model_path, "rb") as file:
    model = pickle.load(file)


st.title("Salary Prediction App 💰")
st.write("Enter your years of experience to predict your salary.")


years = st.number_input("Years", min_value=0, max_value=50, step=1)
monthes = st.number_input("Monthes", min_value=0, max_value=50, step=1)
days = st.number_input("Days", min_value=0, max_value=100, step=5)
experience = years + monthes/12 + days/365
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_pred , y_test)

if st.button("Predict Salary"):
    experience_array = np.array([[experience]], dtype=float)
    predicted_salary = model.predict(experience_array)[0]

    
    st.success(f"Your predicted salary: $ {float(predicted_salary):.2f}")
    
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"📊 Mean Absolute Error (MAE): ${mae:,.2f}")
