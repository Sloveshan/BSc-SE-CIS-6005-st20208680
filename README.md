# Insurance Response Prediction by Sloveshan Dayalan (st20208680)

This project involves building a machine learning application to predict whether a user will respond positively to an insurance offer. The project is divided into several components:

- Streamlit App: A web-based interface for users to input data and get predictions.
- Notebooks: Jupyter notebooks used for model training and evaluation.

## Project Structure

### Frontend Prediction

This folder contains the Streamlit application for predicting insurance responses. To run the application:

- Open the Frontend Prediction folder in Visual Studio Code (VSCode).

- Run the Streamlit app using the command: "streamlit run app.py"

## Notebooks

### Notebook 1: Neural Network Approach

- Purpose: Generates the preprocessing pipeline and trained model used in the Streamlit app.
- Key Components: Exploratory Data Analysis, Data cleaning and balancing, Feature scaling and encoding, Neural network training and evaluation
- Outputs: preprocessing_pipeline.pkl, trained_model.keras

### Notebook 2: Machine Learning Approach

- Purpose: Focuses on traditional machine learning techniques for Kaggle submissions.
- Key Components: Detailed data exploration and cleaning, Feature selection, Hyperparameter tuning using grid search, Model performance evaluation

## How to Use the Streamlit Application

### Input Fields:

- Age: Enter the age of the individual (number greater than or equal to 18).
- Annual Premium: Enter the annual premium amount (numeric value).
- Vintage of the Policy: Enter the vintage of the policy (non-negative number).
- Gender: Select the gender from the dropdown (Male or Female).
- Vehicle Age: Select the vehicle age from the dropdown (< 1 Year, 1-2 Year, > 2 Years).
- Vehicle Damage: Select whether the vehicle has damage (Yes or No).
- Previously Insured: Select whether the individual was previously insured (Yes or No).

### Prediction:

- Click the "Predict" button to get the prediction based on the provided inputs.
- The application will display the predicted response and probability.

## Dependencies

### Ensure you have the following Python packages installed:

- streamlit
- numpy
- pandas
- pickle
- tensorflow
- scikit-learn

Dataset source: https://www.kaggle.com/competitions/playground-series-s4e7/data
