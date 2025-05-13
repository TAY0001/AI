import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import joblib

# --- Functions ---

@st.cache_data
def load_model(model_option):
    """Load the pre-trained model based on user selection."""
    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model = joblib.load('random_forest_model_compressed.joblib')
    elif model_option == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(random_state=42)
        model = joblib.load('gradient_boosting_model.joblib')
    elif model_option == "Naive Bayes":
        model = GaussianNB(priors=[0.5, 0.5])
        model = joblib.load('gaussian_nb_model.joblib')
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model = joblib.load('xgb_classifier_model.joblib')
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    return accuracy, precision, recall, f1, roc_auc, y_pred, y_prob

def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdens_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youdens_j)]
    return best_threshold

# --- Main App ---

# Title
st.title("🏦 Credit Risk Prediction Dashboard")

# Sidebar - Settings
st.sidebar.header("🔍 Model and Input Settings")
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting Classifier", "Naive Bayes", "XGBoost"])

# Load Model
model = load_model(model_option)

# Sidebar - PCA Setting
apply_pca = st.sidebar.checkbox("Apply PCA (Dimensionality Reduction)", value=False)

# PCA Mode (optional: Manual or Auto)
if apply_pca:
    pca_mode = st.sidebar.selectbox("Select PCA Mode", ["Manual", "Auto"])

# Sidebar - Threshold Tuning
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

# Input Form
st.sidebar.header("📝 Input Features")
with st.sidebar.form(key="input_form"):
    person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=18, step=1)
    person_income = st.sidebar.number_input("Income ($)", min_value=0.0, value=0.0)
    person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=5, step=1)
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0.0, value=0.0)
    loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=0.0)
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
    cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=10, step=1)
    submit_button = st.form_submit_button(label="Predict")

# Prepare Input
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

# Scaling
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Apply PCA if selected
if apply_pca:
    if pca_mode == "Manual":
        n_components = st.sidebar.slider("Number of PCA Components", 1, input_data.shape[1], value=2)
        pca = PCA(n_components=n_components)
    else:  # Auto (keep 95% variance)
        pca = PCA(n_components=0.95)
    input_data_scaled = pca.fit_transform(input_data_scaled)

# Prediction
if submit_button:
    probability = model.predict_proba(input_data_scaled)
    prediction = (probability[:,1] >= threshold).astype(int)

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ **Low Risk**")
    else:
        st.error("⚠️ **High Risk**")

    st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
    st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")
    st.write(f"Applied Threshold: **{threshold:.2f}**")

# Display Model Performance - Only needed if model performance is relevant
st.subheader(f"📊 {model_option} Model Performance (Threshold = {threshold:.2f})")
accuracy, precision, recall, f1, roc_auc, _, _ = evaluate_model(model, input_data_scaled, input_data_scaled, threshold)

st.table(pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
}))
