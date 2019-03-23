import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv('full_data.csv')

#x_data = data.loc[:, data.columns != "y"]
#y_data = data.loc[:, "y"]
columns_to_drop = [\
    'Year','StateAbbr', 'DataSource', \
    'Category', 'Measure', \
    'Data_Value_Unit', 'Data_Value_Footnote',\
    'Data_Value_Type', 'Low_Confidence_Limit', \
        'GeoLocation',\
    'High_Confidence_Limit', 'Data_Value_Footnote_Symbol', \
    'CategoryID','Short_Question_Text']

columns_to_keep = [\
    'StateDesc',\
    'CityName',
    'UniqueID', 'GeographicLevel',\
    'DataValueTypeID',\
    'PopulationCount',\
    'CityFIPS','TractFIPS']
    #MeasureId and Data_value are kept but are used to transpose the data

#drop unecessary columns
data = data.drop(columns=columns_to_drop)
#print(data.columns)

prevention_cols = [\
    'ACCESS2', 'BPMED', \
    'CHECKUP', 'CHOLSCREEN', \
    'COLON_SCREEN', 'COREM', \
    'COREW', 'DENTAL', 'MAMMOUSE', 'PAPTEST']

behavior_cols = ['BINGE', 'CSMOKING', 'LPA', 'OBESITY', 'SLEEP']

outcome_cols = [\
    'ARTHRITIS', 'BPHIGH', 'CANCER', 
    'CASTHMA', 'CHD', 'COPD', \
    'DIABETES', 'HIGHCHOL', 'KIDNEY', \
    'MHLTH', 'PHLTH', 'STROKE', 'TEETHLOST']

#remove age-adjusted data
data = data.drop(data[data.DataValueTypeID == 'AgeAdjPrv'].index)
#data = data.drop(data[data.Data_Value == 'AgeAdjPrv'].index)

data = data.pivot_table(index=columns_to_keep,columns = 'MeasureId', values = 'Data_Value')

# Find and plot correlation between outcome features and other features
for outcome in outcome_cols[:]:
    corr_cols = [outcome]
    corr_cols.extend(behavior_cols)
   # corr_cols.extend(prevention_cols)
    cancer_data = data[corr_cols]
    corr = cancer_data.corr()
    print(outcome, ' correlation')
    print(corr[outcome],'\n')
  
    plt.plot(corr[outcome])
    plt.ylabel(outcome)
    plt.xticks(rotation='vertical')
    plt.show()

 #   ax = sns.heatmap(corr)
 #   plt.show()


# Uncomment to export tranposed data
data.to_csv('cleaned_data.csv')
#print(data.columns)