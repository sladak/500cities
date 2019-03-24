import numpy as np
import pandas as pd
import time
import math

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("500_Cities__Local_Data_for_Better_Health__2018_release.csv")
data.head()
columns_to_drop = [
    'Year', 'StateAbbr', 'DataSource', 'Measure',
    'Data_Value_Unit', 'Data_Value_Footnote',
    'Data_Value_Type', 'Low_Confidence_Limit',
    # 'GeoLocation',\
    'High_Confidence_Limit', 'Data_Value_Footnote_Symbol',
    'CategoryID', 'Short_Question_Text']

columns_to_keep = ['StateDesc', 'Category', 'CityName', 'UniqueID', 'GeographicLevel', 'DataValueTypeID',
                   'PopulationCount', 'CityFIPS', 'TractFIPS', 'GeoLocation']
# MeasureId and Data_value are kept but are used to transpose the data

# drop unecessary columns
data = data.drop(columns=columns_to_drop)
# print(data.columns)

prevention_cols = [
    'ACCESS2', 'BPMED',
    'CHECKUP', 'CHOLSCREEN',
    'COLON_SCREEN', 'COREM',
    'COREW', 'DENTAL', 'MAMMOUSE', 'PAPTEST']

behavior_cols = ['BINGE', 'CSMOKING', 'LPA', 'OBESITY', 'SLEEP']

outcome_cols = [
    'ARTHRITIS', 'BPHIGH', 'CANCER',
    'CASTHMA', 'CHD', 'COPD',
    'DIABETES', 'HIGHCHOL', 'KIDNEY',
    'MHLTH', 'PHLTH', 'STROKE', 'TEETHLOST']

# remove age-adjusted data
data = data.drop(data[data.DataValueTypeID == 'AgeAdjPrv'].index)
# data = data.drop(data[data.Data_Value == 'AgeAdjPrv'].index)

census_tract_data = data[data['GeographicLevel'] == 'Census Tract']

city_data = data[data['GeographicLevel'] == 'City']
print("Size of census tract data:", len(census_tract_data))

tract_pv = census_tract_data.pivot_table(index=['CityName', 'UniqueID'], columns='MeasureId', values='Data_Value',
                                         aggfunc='sum')
print("Size of census tract data:", len(ct_pv))
tract_pv = tract_pv.fillna(tract_pv.mean())

city_pv = census_tract_data.pivot_table(index=['CityName'], columns='MeasureId', values='Data_Value', aggfunc='sum')
print("Size of city data:", len(city_pv))
city_pv = city_pv.fillna(city_pv.mean())

city_pv.reset_index(level=0, inplace=True)
tract_pv.reset_index(level=0, inplace=True)
