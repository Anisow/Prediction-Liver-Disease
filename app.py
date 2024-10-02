import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Prediction of Liver Disease", layout="wide")

# Load the model and feature names
model_path = 'models/best_rf_model.pkl'  # Update with your model path

try:
    with open(model_path, 'rb') as file:
        loaded_model, feature_names = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model path is correct.")
    st.stop()

# Streamlit app title
st.title('Prediction of Liver Disease')

st.write("## Enter the patient information:")

# Input fields for all features based on your dataset
def user_input_features():
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    tot_bilirubin = st.number_input('Total Bilirubin (mg/dL)', min_value=0.0, value=0.5)
    direct_bilirubin = st.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, value=0.2)
    tot_proteins = st.number_input('Total Proteins (g/dL)', min_value=0.0, value=6.0)
    albumin = st.number_input('Albumin (g/dL)', min_value=0.0, value=3.0)
    ag_ratio = st.number_input('A/G Ratio', min_value=0.0, value=1.0)
    sgpt = st.number_input('SGPT (U/L)', min_value=0.0, value=20.0)
    sgot = st.number_input('SGOT (U/L)', min_value=0.0, value=20.0)
    alkphos = st.number_input('Alkaline Phosphatase (U/L)', min_value=0.0, value=60.0)

    # Prepare input data
    data = {
        'age': age,
        'gender': 1 if gender == 'Male' else 0,  # Convert to numerical
        'tot_bilirubin': tot_bilirubin,
        'direct_bilirubin': direct_bilirubin,
        'tot_proteins': tot_proteins,
        'albumin': albumin,
        'ag_ratio': ag_ratio,
        'sgpt': sgpt,
        'sgot': sgot,
        'alkphos': alkphos
    }
    
    features = pd.DataFrame(data, index=[0])
    display_data = {
        'Age': age,
        'Gender': gender,
        'Total Bilirubin': tot_bilirubin,
        'Direct Bilirubin': direct_bilirubin,
        'Total Proteins': tot_proteins,
        'Albumin': albumin,
        'A/G Ratio': ag_ratio,
        'SGPT': sgpt,
        'SGOT': sgot,
        'Alkaline Phosphatase': alkphos
    }
    return features, display_data

# Get user input
input_df, display_data = user_input_features()

# Display the user input
st.write("### Patient's Information:")
st.write(pd.DataFrame(display_data, index=[0]))

# Prediction button
if st.button('Predict'):
    # Make predictions
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    # Display the prediction
    st.write("### Prediction")
    if prediction[0] == 1:
        st.write("**Result:** Patient has liver disease.")
    else:
        st.write("**Result:** Patient is healthy.")

    # Display the prediction probabilities
    st.write("### Prediction Probability")
    st.write(f"Probability of having liver disease: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Probability of not having liver disease: **{prediction_proba[0][0]:.2f}**")
