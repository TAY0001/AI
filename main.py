import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_risk.joblib')

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Collect important features in the sidebar
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
person_income = st.sidebar.number_input("Income ($)", min_value=0.0, value=50000.0)
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=5, step=1)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)
loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.sidebar.number_input("Loan Percent Income (%)", min_value=0.0, max_value=100.0, value=20.0)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=10, step=1)

# Optional features with default values
st.sidebar.subheader("Optional Features")
optional_features = {
    'person_home_ownership_RENT': 0,
    'person_home_ownership_OWN': 0,
    'cb_person_default_on_file_Y': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_EDUCATION': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0,
    'person_home_ownership_OTHER': 0
}

if st.sidebar.checkbox("Include Home Ownership (RENT)"):
    optional_features['person_home_ownership_RENT'] = st.sidebar.number_input("Home Ownership (RENT) (0 or 1)", min_value=0, max_value=1, value=0)
if st.sidebar.checkbox("Include Home Ownership (OWN)"):
    optional_features['person_home_ownership_OWN'] = st.sidebar.number_input("Home Ownership (OWN) (0 or 1)", min_value=0, max_value=1, value=0)
if st.sidebar.checkbox("Include Default on File (Y)"):
    optional_features['cb_person_default_on_file_Y'] = st.sidebar.number_input("Default on File (Y) (0 or 1)", min_value=0, max_value=1, value=0)
if st.sidebar.checkbox("Include Loan Intent"):
    loan_intent_options = ['HOMEIMPROVEMENT', 'MEDICAL', 'EDUCATION', 'PERSONAL', 'VENTURE']
    selected_intent = st.sidebar.selectbox("Select Loan Intent", loan_intent_options)
    for intent in loan_intent_options:
        optional_features[f'loan_intent_{intent}'] = 1 if intent == selected_intent else 0

# Prepare features for prediction
important_features = [
    loan_percent_income,
    loan_int_rate,
    person_income,
    loan_amnt,
    person_emp_length,
    person_age,
    cb_person_cred_hist_length
]

# Combine important and optional features
feature_inputs = important_features + list(optional_features.values())

# Prediction
if st.sidebar.button("Predict Credit Risk"):
    features_array = np.array([feature_inputs])
    try:
        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("High Risk")
        else:
            st.success("Low Risk")
        
        st.subheader("Prediction Probabilities")
        st.write(f"Low Risk Probability: {probabilities[0][0] * 100:.2f}%")
        st.write(f"High Risk Probability: {probabilities[0][1] * 100:.2f}%")
        
    except ValueError as e:
        st.error(f"Prediction failed: {e}")

st.markdown("\n**Note:** This is a demo app. The prediction is based on the input features and model logic.")
