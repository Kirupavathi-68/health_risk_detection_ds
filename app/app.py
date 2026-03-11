
import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib
import joblib

# Load trained model and X_train
model = joblib.load("../notebooks/logistic_model.pkl")
X_train = joblib.load("../notebooks/X_train.pkl")



st.title("Heart Disease Risk Prediction")

# Take inputs
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120?", [0,1])
restecg = st.number_input("Resting ECG (0-2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina?", [0,1])
oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
thal = st.number_input("Thal (1-3)", 1, 3, 2)

# Predict
input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]],
                        columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                 'thalach','exang','oldpeak','slope','ca','thal'])

prediction = model.predict(input_df)[0]
st.write("Heart Disease Risk:", "Yes ❤️" if prediction==1 else "No 💚")

if st.button("Explain Prediction"):
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(input_df)
    st.write("Feature importance for this prediction:")
    shap.initjs()
    st.pyplot(shap.summary_plot(shap_values, input_df))

