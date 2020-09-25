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
    prob_drop = round(prob_drop,2)*100
    prob_drop = prob_drop["prob_drop"][0]
    percent = "%"
    prob_drop = str(prob_drop) + percent
    return prob_drop

def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.title("Centerstone's Research Institute Evaluation Dropout Predictor")
   
    if add_selectbox == 'Online':
        st.subheader('Please answer the following questions using the scale below: \n \
        1 = strongly disagree \n \
        2 = disagree \n \
        3 = undecided \n \
        4 = agree \n \
        5 = strongly agree')
        HandlingDailyLife = st.number_input('I deal effectively with daily problems.', min_value=1, max_value=5, value= 1)
        SocialSituations = st.number_input('I do well in social situations.', min_value=1, max_value=5, value=1)
        FunctioningHousing = st.number_input('My housing situation is satisfactory.', min_value=1, max_value=5, value=1)
        Symptoms = st.number_input('My symptoms are not bothering me.', min_value=1, max_value=5, value=1)
        Friendships = st.number_input('I am happy with the friendships I have.', min_value=1, max_value=5, value=1)
        EnjoyPeople = st.number_input('I have people with whom I can do enjoyable things.', min_value=1, max_value=5, value=1)
        BelongInCommunity = st.number_input('I feel I belong in my community.', min_value=1, max_value=5, value=1)
        RelationshipSatisfaction = st.number_input('In the last 4 weeks, how satisfied are you with\
        your personal relationships? \n \
        1 = very dissatisfied\n \
        2 = dissatisfied \n \
        3 = neither satisfied nor dissatisfied \n \
        4 = satisfied \n \
        5 = satisfied', min_value=1, max_value=5, value=1)
        st.subheader("For questions about tobacco, alcohol, and cannabis use, please use the scale below: \n \
        1 = never \n \
        2 = once or twice \n \
        3 = weekly \n \
        4 = daily or almost daily")
        Tobacco_Use = st.number_input('In the past 30 days, how often have you used… \
        tobacco products (cigarettes, chewing tobacco, cigars, etc.)?', min_value=1, max_value=4, value=1)
        Alcohol_Use = st.number_input('In the past 30 days, how often have you used… \
        alcoholic beverages (beer, wine, liquor, etc.)?', min_value=1, max_value=4, value=1)
        Cannabis_Use = st.number_input('In the past 30 days, how often have you used… \
        cannabis (marijuana, pot, grass, hash, etc.)?', min_value=1, max_value=4, value=1)
        ViolenceTrauma = st.number_input('Have you ever experienced violence or trauma in any setting (including community or school violence; domestic violence; physical, psychological, or \
        sexual maltreatment/assault within or outside of the family; natural disaster; \
        terrorism; neglect; or traumatic grief)? \n \
        1 = yes \n \
        0 = no', min_value=0, max_value=1, value=1)
        st.subheader("For the education question below, please use the scale below: \n \
        11 = less than 12th grade \n \
        12 = 12th grade / high school diploma / equivalent (GED) \n \
        13 = voc/tech diplomia \n \
        14 = some college or university \n \
        15 = bachelor's degree (BA, BS) \n \
        16 = graduate work / graduate degree")
        Education = st.number_input('Education', min_value=11, max_value=16, value=11)
        EnoughMoneyForNeeds = st.number_input('In the last 4 weeks, have you enough money to meet your needs? 1 = not a all \
        2 = a little 3 = moderatly 4 = mostly 5 = completely', min_value=1, max_value=5, value=1)
        grant = st.number_input('Is your client enrolled in CCBHC 1= yes 0 = no', min_value=1, max_value=5, value=1)
        EverServed = st.number_input('Have you ever served in the Armed Forces, \
        the Reserves, or the National Guard? 1 = yes 0 = no', min_value=0, max_value=1, value=1)
        mdd_r = st.number_input('Is the client diagnosed with Major Depression Disorder that is recurrent? 1 = yes 0 = no', min_value=0, max_value=1, value=1)
        another_s_ident = st.number_input('Does the client identify as a sexual orientation that is heterosexual? 1 = yes 0 = no', min_value=1, max_value=5, value=1)
        drug_use = st.number_input('In the past 30 days, how often have you used any substance other than \
        alcohol, cannabis, and tobacco', min_value=1, max_value=30, value=1)
        er_hos_use_base = st.number_input('In the past 30 days, how many nights have you spent in the \
        hosptial and how many times have you visited emergancy room for mental health care', min_value=1, max_value=30, value=1)
       
        output=""
        input_dict = {'f0' : HandlingDailyLife, 'f1': SocialSituations, "f2": FunctioningHousing, "f3": Symptoms, "f4": RelationshipSatisfaction, "f5": Tobacco_Use, "f6": Alcohol_Use, "f7": Cannabis_Use, "f8": ViolenceTrauma, "f9": Education, "f10": EnoughMoneyForNeeds, "f11": Friendships, "f12": EnjoyPeople, "f13": BelongInCommunity, "f14": grant, "f15": EverServed, "f16": mdd_r, "f17": another_s_ident, "f18": drug_use, "f19": er_hos_use_base}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(input_df=input_df)
            st.success('Probability of not getting a 6-month reassessment is {}'.format(output,2))
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
