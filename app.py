#!/usr/bin/env python
# coding: utf-8

# In[23]:

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

filename = "best_model_9_14_20_3pm.pkl"
best_model = joblib.load(filename)

def predict(input_df):
    predictions_df = best_model.predict_proba(input_df)
    prob_drop =   pd.DataFrame(predictions_df[:,1], columns = ["prob_drop"])
    return prob_drop

def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.title("Insurance Charges Prediction App")
    if add_selectbox == 'Online':
        #Quarter	Gender	   could be f0, f1
        Quarter = st.number_input('Quarter', min_value=1, max_value=4, value=1)
        Gender = st.number_input('Gender', min_value=0, max_value=1, value=1)
        output=""
        input_dict = {'f0' : Quarter, 'f1': Gender}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(input_df=input_df)
            st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions_df = best_model.predict_proba(data)
            st.write(predictions)

if __name__ == '__main__':
    run()
