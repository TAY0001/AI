import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import joblib



# Load dataset
model = joblib.load('credit_risk (1).joblib')

# Split the data into features and target
X = df.drop(columns=['loan_status'], axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the function for model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, roc_auc

# Streamlit app layout
st.title("Model Comparison Dashboard")

# Model selection
model_option = st.selectbox("Select a model", ["Random Forest", "SVM", "Naive Bayes"])

# Feature input for prediction
st.subheader("Enter the features for prediction")

# List of features based on dataset columns (adjust based on your dataset)
person_age = st.number_input("Person Age", min_value=0, max_value=100, value=18)
person_income = st.number_input("Person Income", min_value=0, value=0)
person_emp_length = st.number_input("Person Employment Length (in years)", min_value=0, max_value=50, value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=0)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=0.0)
if person_income > 0:
    loan_percent_income = (loan_amnt / person_income) * 100
else:
    loan_percent_income = 0.0

st.sidebar.number_input(
    "Loan Percent Income (%)",
    value=loan_percent_income,
    format="%.2f",
    disabled=True
)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=0, step=1)

# Store the input data into a DataFrame
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

if prediction[0] == 0:
    st.write("Prediction: Low Risk")
else:
    st.write("Prediction: High Risk")
