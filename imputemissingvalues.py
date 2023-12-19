# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:37:46 2021

@author: Johnny Feng
"""

# missing values imputation

import datawig
from datawig import Imputer
from datawig.column_encoders import *
from datawig.mxnet_input_symbols import *
from datawig.utils import random_split
import pandas as pd


# import data
data = pd.read_csv("E:\\Boston University\\MA679\\FinalProj\\Transformdata_datawig.csv")
raw_data = pd.read_csv("E:\\Boston University\\MA679\\FinalProj\\Rawdata_datawig.csv")
df_train, df_test = random_split(data, split_ratios=[0.8, 0.2])

#Specify encoders and featurizers
imputer_1 = datawig.SimpleImputer(
    input_columns=['Radiation','Chemotherapy','Cause.of.Death','Age.at.Diagnosis','Surgery.Performed.','Site'], # column(s) containing information about the column we want to impute
    output_column= 'AJCC.7.Stage', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer_1.fit(train_df=df_train, num_epochs=50)


#Specify encoders and featurizers
imputer_2 = datawig.SimpleImputer(
    input_columns=['Radiation','Chemotherapy','Cause.of.Death','Age.at.Diagnosis','Surgery.Performed.','Site'], # column(s) containing information about the column we want to impute
    output_column= 'Insurance', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer_2.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
# imputed_2 = imputer_2.predict(df_test)

imputer_3 = datawig.SimpleImputer(
    input_columns=['Radiation','Chemotherapy','Cause.of.Death','Age.at.Diagnosis','Surgery.Performed.','Site'], # column(s) containing information about the column we want to impute
    output_column= 'Lymph.Nodes', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer_3.fit(train_df=df_train, num_epochs=50)

imputer_4 = datawig.SimpleImputer(
    input_columns=['Radiation','Chemotherapy','Cause.of.Death','Age.at.Diagnosis','Surgery.Performed.','Site'], # column(s) containing information about the column we want to impute
    output_column= 'Mets', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer_4.fit(train_df=df_train, num_epochs=50)

imputer_5 = datawig.SimpleImputer(
    input_columns=['Radiation','Chemotherapy','Cause.of.Death','Age.at.Diagnosis','Surgery.Performed.','Site'], # column(s) containing information about the column we want to impute
    output_column= 'T_Stage', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer_5.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer_1.predict(raw_data)
imputed = imputer_2.predict(imputed)
imputed = imputer_3.predict(imputed)
imputed = imputer_4.predict(imputed)
imputed = imputer_5.predict(imputed)
# imputed.drop(imputed.columns[0], axis=1, inplace=True)

imputed.to_csv("E:\\Boston University\\MA679\\FinalProj\\imputed.csv",index=False,sep=',')
