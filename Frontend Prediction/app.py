import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.exceptions import NotFittedError

# Load the preprocessing pipeline from a pickle file
with open('preprocessing_pipeline.pkl', 'rb') as f:
    prep = pickle.load(f)

# Test the preprocessing pipeline to ensure it's fitted
try:
    prep.transform(pd.DataFrame([{
        'Age': 25, 
        'Annual_Premium': 30000, 
        'Vintage': 150, 
        'Gender': 'Male', 
        'Vehicle_Age': '< 1 Year', 
        'Vehicle_Damage': 'Yes', 
        'Previously_Insured': 'No'
    }]))
except NotFittedError:
    print("The transformer is not fitted and 'train.csv' is not being loaded.")

# Load the pre-trained model
model = tf.keras.models.load_model('trained_model.keras')
print("Model loaded successfully.")

# Streamlit app title
st.title('Insurance Response Prediction')

# Create two columns in the UI for input fields
col1, col2 = st.columns(2)

with col1:
    # Age input field
    age_input = st.text_input('Age', '25')

# Validate the age input to ensure it's a number and at least 18
valid_age = age_input.isdigit() and int(age_input) >= 18
if valid_age:
    age = int(age_input)
else:
    st.error("Please enter a valid age (number greater than or equal to 18).")

with col2:
    # Annual Premium input field
    annual_premium_input = st.text_input('Annual Premium', '30000')

# Validate the annual premium input to ensure it's numeric
if annual_premium_input.isdigit():
    annual_premium = int(annual_premium_input)
else:
    st.error("Please enter a valid annual premium (numeric value).")
    annual_premium = 30000  # Default value if input is invalid

# Create two more columns for additional input fields
col3, col4 = st.columns(2)

with col3:
    # Vintage input field (how long the policy has been held)
    vintage_input = st.text_input('Vintage of the Policy', '150')

# Validate the vintage input to ensure it's a non-negative number
if vintage_input.isdigit() and int(vintage_input) >= 0:
    vintage = int(vintage_input)
else:
    st.error("Please enter a valid vintage (a non-negative number).")
    vintage = 0  # Default value if input is invalid

with col4:
    # Gender input field as a dropdown
    gender = st.selectbox('Gender', options=['Male', 'Female'])

# Create two more columns for the remaining input fields
col5, col6 = st.columns(2)

with col5:
    # Vehicle Age input field as a dropdown
    vehicle_age = st.selectbox('Vehicle Age', options=['< 1 Year', '1-2 Year', '> 2 Years'])

with col6:
    # Vehicle Damage input field as a dropdown
    vehicle_damage = st.selectbox('Vehicle Damage', options=['Yes', 'No'])

# Previously Insured input field as a dropdown
previously_insured = st.selectbox('Previously Insured', options=['Yes', 'No'])

# Create a dictionary to hold all the input data
input_data = {
    'Age': age if valid_age else None,  # Age is included only if valid
    'Annual_Premium': annual_premium,
    'Vintage': vintage,
    'Gender': gender,
    'Vehicle_Age': vehicle_age,
    'Vehicle_Damage': vehicle_damage,
    'Previously_Insured': previously_insured
}

# Check if all inputs are valid (i.e., none are None)
all_inputs_valid = all(value is not None for value in input_data.values())

# Convert the input data dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Create a container to align the predict button and prediction output
with st.container():
    col3, col4 = st.columns([2, 1])

    with col3:
        # Predict button; only enabled if all inputs are valid
        if st.button('Predict', disabled=not all_inputs_valid):
            # Preprocess the input data using the loaded pipeline
            input_transformed = prep.transform(input_df)

            # Make predictions using the loaded model
            prediction = model.predict(input_transformed)

            # Extract the predicted probability
            predicted_prob = prediction[0][0]

            # Determine the response based on the predicted probability
            response = 'Yes' if predicted_prob > 0.5 else 'No'

            # Display the prediction with both the response and the probability
            st.markdown(f"""
            <p style='text-align: left; font-weight: bold; margin:-60px 0px 0px 100px;'>
            Predicted Response: {response} <br>
            Predicted Probability: {predicted_prob:.2f}
            </p>
            """, unsafe_allow_html=True)

# Add a caption to explain the prediction output
st.caption("""
This app predicts whether a user is likely to respond positively to an insurance offer based on their input data.

**"Yes"** means this could indicate that the user is likely to purchase or show interest in the insurance product being offered.

**"No"** means this suggests that the user may not be interested in purchasing or engaging with the insurance product.

If prediction[0][0] is greater than 0.5, it outputs "Yes" (indicating a positive prediction, such as the customer likely buying a product). Otherwise, it outputs "No."
""")
