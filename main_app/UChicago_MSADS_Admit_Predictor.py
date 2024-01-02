#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")
import os

# # Data Generation

# In[2]:
np.random.seed(42)
num_samples = 10000  # number of samples cane be increased as needed
data = {
    'Undergraduate_GPA': np.random.randint(3,4, num_samples),
    'Highest_degree':np.random.choice(['Bachelors','Masters','Other_three_year_degree'], num_samples),
    'Relevant_DataScience_Experience ': np.random.uniform(0,8, num_samples),
    'Degree_seeking_Type': np.random.choice(['Masters','PHD'], num_samples),
    'Apply_Year': np.random.randint(2020,2023, num_samples),
    'Country': np.random.choice(['United States','Canada','Asia','Europe','India'], num_samples),
    'Uploaded_Statement_of_Purpose_Status': np.random.choice(['Y', 'N'], num_samples),
    'Uploaded_Resume_Status': np.random.choice(['Y', 'N'], num_samples),  # Assuming the candidate is confident of his/her abilities
    'GRE_Score': np.random.randint(300,325, num_samples),
    'IELTS_Overall': np.random.randint(5,10, num_samples),
    'TOEFL_Overall': np.random.choice([90,100], num_samples),
    'Gender': np.random.choice(['M','F'], num_samples),
    'DataScience_Skill_Confidence': np.random.randint(0,10, num_samples),
    'Admit_label': np.random.choice([0, 1], num_samples)  # Binary target variable (approved or declined)
}

df = pd.DataFrame(data)
# # Train the Model 

# In[3]:
# Separate features and labels
X = df.drop('Admit_label', axis=1)
y = df['Admit_label']
# Create dummy variables for categorical features
X = pd.get_dummies(X, columns=['Degree_seeking_Type', 'Highest_degree','Country', 'Uploaded_Statement_of_Purpose_Status', 'Uploaded_Resume_Status', 'Gender'], drop_first=True)
df = pd.concat([df, X],axis=1)
df = df.drop(columns={'Degree_seeking_Type','Highest_degree','Country', 'Uploaded_Statement_of_Purpose_Status', 'Uploaded_Resume_Status', 'Gender'},axis=1)
df = df.T.drop_duplicates().T
# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Train XGBoostClassifier
clf_xgb = XGBClassifier(n_estimators=100, random_state=42)
clf_xgb.fit(X_train,y_train)
# Evaluate the model on the validation set
y_val_pred = clf_xgb.predict(X_val)
# Display evaluation metrics
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))
# Save the trained XGBoostClassifier model
pickle.dump(clf_xgb, open('xgb_Admit_predictor_model.sav', 'wb'))
# Save dummy data for reference
df.to_csv('sample_Admit_predictor_model_training_data.csv', index=False)


# # Lets Deploy the Base iteration model in the Streamlit app 

# In[7]:
    
def user_input_features():
    apply_year = st.number_input("Apply_Year",min_value=2024, max_value=2025)
    undergraduate_gpa = st.number_input("Undergraduate_GPA",min_value=3.0, max_value=4.0,step = 0.1)
    gender = st.selectbox("Gender", ('M', 'F'))
    resume_status = st.selectbox("Uploaded_Resume_Status", ('Y', 'N'))
    sop_status = st.selectbox("Uploaded_Statement_of_Purpose_Status", ('Y', 'N'))
    relevant_datascience_experience = st.number_input("Relevant_DataScience_Experience", min_value=0, max_value=8, step=1)
    gre_score = st.number_input("GRE_Score", min_value=300, max_value=325, step=1)
    ielts_overall = st.number_input("IELTS_Overall", min_value=5.0, max_value=10.0, step = 0.5)
    toefl_overall = st.number_input("TOEFL_Overall", min_value=90, max_value=110, step=1)
    datascience_skill_confidence = st.number_input("DataScience_Skill_Confidence", min_value=0, max_value=10, step=1)
    degree_seeking_type = st.selectbox("Degree_seeking_Type",('Masters','PHD'))
    highest_degree = st.selectbox("Highest_degree",('Bachelors','Masters','Other_three_year_degree'))
    country = st.selectbox("Country",('United States','Canada','Asia','Europe','India'))
    data = {'Apply_Year': apply_year,
            'Undergraduate_GPA': undergraduate_gpa,
            'Gender': gender,
            'Uploaded_Resume_Status': resume_status,
            'Uploaded_Statement_of_Purpose_Status': sop_status,
            'Relevant_DataScience_Experience': relevant_datascience_experience,
            'GRE_Score': gre_score,
            'IELTS_Overall': ielts_overall,
            'TOEFL_Overall': toefl_overall,
            'DataScience_Skill_Confidence': datascience_skill_confidence,
            'Degree_seeking_Type': degree_seeking_type,
            'Highest_degree': highest_degree,
            'Country': country,
           }
    features = pd.DataFrame(data, index=[0])
    return features

def run_app(clf_xgb):
    st.write("""
    # Admit Predictor for MS in Applied Data Science  
    This app generously predicts the probability of admit based on custom input parameters
    
    [*Click here to view the code for the app*](https://github.com/NANDANKESHAVHEGDE/Learn_Streamlit_app_building)
    """)
    # Get Input
    st.header('User Input Parameters')
    df_input = user_input_features()
    st.subheader('User Input parameters')
    st.write(df_input.to_dict())
    # Create dummy variables for categorical features
    cat_features = ['Degree_seeking_Type','Highest_degree','Country', 'Uploaded_Statement_of_Purpose_Status', 'Uploaded_Resume_Status', 'Gender']
    df_input_encoded = pd.get_dummies(df_input, columns=cat_features, drop_first=True)
    # Add missing columns with default values
    missing_columns = set(clf_xgb.get_booster().feature_names) - set(df_input_encoded.columns)
    for col in missing_columns:
        df_input_encoded[col] = 0
    # Reorder columns to match the model's feature names
    df_input_encoded = df_input_encoded[clf_xgb.get_booster().feature_names]
    # Model Loading
    clf_xgb = pickle.load(open('xgb_Admit_predictor_model.sav', 'rb'))
    # Model Inferencing
    prediction = clf_xgb.predict(df_input_encoded)
    prediction_proba = clf_xgb.predict_proba(df_input_encoded)
    st.subheader('Prediction')
    if prediction == 0:
        st.write('Declined')
    else:
        st.write('Admitted')
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if __name__ == '__main__':
    run_app(clf_xgb)

