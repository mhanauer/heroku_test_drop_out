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

filename = "gs_9_24_20_1pm.joblib"
best_model = joblib.load(filename)


def predict(input_df):
    predictions_df = best_model.predict_proba(input_df)
    prob_drop =   pd.DataFrame(predictions_df[:,1], columns = ["prob_drop"])
    return prob_drop

def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.title("Centerstone's Research Institute Treatment Dropout Predictor")
    if add_selectbox == 'Online':
        #Quarter	Gender	   could be f0, f1
        HandlingDailyLife = st.number_input('HandlingDailyLife', min_value=1, max_value=5, value=1)
        SocialSituations = st.number_input('SocialSituations', min_value=1, max_value=5, value=1)
        FunctioningHousing = st.number_input('FunctioningHousing', min_value=1, max_value=5, value=1)
        Symptoms = st.number_input('Symptoms', min_value=1, max_value=5, value=1)
        RelationshipSatisfaction = st.number_input('RelationshipSatisfaction', min_value=1, max_value=5, value=1)
        Tobacco_Use = st.number_input('Tobacco_Use', min_value=1, max_value=5, value=1)
        Alcohol_Use = st.number_input('Alcohol_Use', min_value=1, max_value=5, value=1)
        Cannabis_Use = st.number_input('Cannabis_Use', min_value=1, max_value=5, value=1)
        ViolenceTrauma = st.number_input('ViolenceTrauma', min_value=1, max_value=5, value=1)
        Education = st.number_input('Education', min_value=1, max_value=5, value=1)
        EnoughMoneyForNeeds = st.number_input('EnoughMoneyForNeeds', min_value=1, max_value=5, value=1)
        Friendships = st.number_input('Friendships', min_value=1, max_value=5, value=1)
        EnjoyPeople = st.number_input('EnjoyPeople', min_value=1, max_value=5, value=1)
        BelongInCommunity = st.number_input('BelongInCommunity', min_value=1, max_value=5, value=1)
        grant = st.number_input('grant', min_value=1, max_value=5, value=1)
        EverServed = st.number_input('EverServed', min_value=1, max_value=5, value=1)
        mdd_r = st.number_input('mdd_r', min_value=1, max_value=5, value=1)
        another_s_ident = st.number_input('another_s_ident', min_value=1, max_value=5, value=1)
        drug_use = st.number_input('drug_use', min_value=1, max_value=30, value=1)
        er_hos_use_base = st.number_input('er_hos_use_base', min_value=1, max_value=30, value=1)
        
    
        output=""
        input_dict = {'f0' : HandlingDailyLife, 'f1': SocialSituations, "f2": FunctioningHousing, "f3": Symptoms, "f4": RelationshipSatisfaction, "f5": Tobacco_Use, "f6": Alcohol_Use, "f7": Cannabis_Use, "f8": ViolenceTrauma, "f9": Education, "f10": EnoughMoneyForNeeds, "f11": Friendships, "f12": EnjoyPeople, "f13": BelongInCommunity, "f14": grant, "f15": EverServed, "f16": mdd_r, "f17": another_s_ident, "f18": drug_use, "f19": er_hos_use_base}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(input_df=input_df)
            st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            input_df = data.replace([-99, -98, -1, -2, -3, -4, -5, -6, -7, -8, -9], np.nan)
            filter_input_df = input_df["InterviewType_07"] == 1
            input_df = input_df[filter_input_df]
            input_df = input_df.reset_index(drop = True)
            predictors =  input_df
            predictors = predictors[[
            'HandlingDailyLife',
            'SocialSituations',
            'FunctioningHousing',
            'Symptoms',
            'RelationshipSatisfaction',
            'Tobacco_Use',
            'Alcohol_Use',
            'Cannabis_Use',
            'ViolenceTrauma',
            'Education',
            'EnoughMoneyForNeeds',
            'Friendships',
            'EnjoyPeople',
            'BelongInCommunity',
            'EverServed']]
            drug_use = input_df["Cocaine_Use"] +input_df["Meth_Use"] + input_df["StreetOpioids_Use"] +  input_df["RxOpioids_Use"] + input_df["Stimulants_Use"]  + input_df["Inhalants_Use"] +  input_df["Sedatives_Use"] + input_df["Hallucinogens_Use"] + input_df["Other_Use"]
            drug_use = pd.DataFrame(data = drug_use, columns = ["drug_use"])
            drug_use = drug_use.reset_index(drop = True)

            er_hos_use_base = input_df["NightsDetox"] + input_df["NightsHospitalMHC"] + input_df["TimesER"]
            er_hos_use_base = pd.DataFrame(data = er_hos_use_base,columns = ["er_hos_use_base"])
            er_hos_use_base = er_hos_use_base.reset_index(drop = True)
            x = np.array([1])
            grant = np.repeat(x, [len(input_df)], axis=0)
            ## Index 
            grant = pd.DataFrame(data = grant, columns = ["grant"])
            grant = grant.reset_index(drop = True)
            def if_else(row):

                if row['DiagnosisOne'] == 59:
                    val= 1
                else:

                    val = 0

                return val
            mdd_r =  input_df.apply(if_else, axis=1)
            mdd_r = pd.DataFrame(data = mdd_r, columns = ["mdd_r"])
            mdd_r = mdd_r.reset_index(drop = True)

            def if_else(row):

                if row['SexualIdentity'] > 1:
                    val= 1
                else:

                    val = 0

                return val

            another_s_ident =  input_df.apply(if_else, axis=1)
            another_s_ident = pd.DataFrame(data = another_s_ident, columns = ["another_s_ident"])
            another_s_ident = another_s_ident.reset_index(drop = True)

            frames =  [predictors, grant, mdd_r, another_s_ident,drug_use , er_hos_use_base]
            predictors_all = pd.concat(frames, axis = 1)
    
            predictors_all.rename(columns = {'f0': "HandlingDailyLife", 'f1': "SocialSituations", "f2": "FunctioningHousing", "f3": "Symptoms", "f4": "RelationshipSatisfaction", "f5": "Tobacco_Use", "f6": "Alcohol_Use", "f7": "Cannabis_Use", "f8": "ViolenceTrauma", "f9": "Education", "f10": "EnoughMoneyForNeeds", "f11": "Friendships", "f12": "EnjoyPeople", "f13": "BelongInCommunity", "f14": "grant", "f15": "EverServed", "f16": "mdd_r", "f17": "another_s_ident", "f18": "drug_use", "f19": "er_hos_use_base"}, inplace=True)
            predictors_all = predictors_all.to_numpy()
            prob_drop =  best_model.predict_proba(predictors_all)

            prob_drop =   pd.DataFrame(prob_drop[:,1], columns = ["prob_drop"])
            def if_else(row):

                if row['prob_drop'] > 0.65708125:

                   val = "very high risk"

                elif row['prob_drop'] > 0.65708125 / 2:

                    val = "high risk "
    
                elif row['prob_drop'] > 0.65708125 / 3:
        
                    val = "medium risk"
    
                else:

                    val = "low risk"

                return val

            prob_drop['risk_level'] = prob_drop.apply(if_else, axis=1)
            pred_dat = input_df.reset_index(drop = True)
            pred_dat_filter = pred_dat["InterviewType_07"] == 1
            pred_dat = pred_dat[pred_dat_filter]
            ConsumerID =  pred_dat.iloc[:,0]
            drop_out_risk_level = prob_drop.iloc[:,1]
            frames = [ConsumerID, drop_out_risk_level]
            pred_dat = pd.concat(frames, axis = 1)
            st.write(pred_dat)
    st.set_option('deprecation.showfileUploaderEncoding', False)

if __name__ == '__main__':
    run()
