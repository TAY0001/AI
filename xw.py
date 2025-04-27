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

# Allow user to upload dataset through Streamlit's file uploader
uploaded_file = st.file_uploader("Upload Credit Risk Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv("credit_risk_dataset.csv")
else:
    st.warning("Please upload a dataset")

X = df.drop(columns=['loan_status'], axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, roc_auc

st.title("Model Comparison Dashboard")

model_option = st.selectbox("Select a model", ["Random Forest", "SVM", "Naive Bayes"])

st.subheader("Enter the features for prediction")

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

# Based on the selection, train and evaluate the model
if model_option == "Random Forest":
    st.subheader("Random Forest Model")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    accuracy, precision, recall, f1, roc_auc = evaluate_model(rf_model, X_test, y_test)
    prediction = rf_model.predict(input_data_scaled)

elif model_option == "SVM":
    st.subheader("SVM Model")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    accuracy, precision, recall, f1, roc_auc = evaluate_model(svm_model, X_test, y_test)
    prediction = svm_model.predict(input_data_scaled)

elif model_option == "Naive Bayes":
    st.subheader("Naive Bayes Model")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    accuracy, precision, recall, f1, roc_auc = evaluate_model(nb_model, X_test, y_test)
    prediction = nb_model.predict(input_data_scaled)

# Display the evaluation results
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")
st.write(f"ROC AUC Score: {roc_auc:.4f}")

# Display prediction result
if prediction[0] == 0:
    st.write("Prediction: Low Risk")
else:
    st.write("Prediction: High Risk")

# Display confusion matrix as heatmap
cm = confusion_matrix(y_test, prediction)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
