prevention_cols = [
    'ACCESS2', 'BPMED', 'CHECKUP', 'CHOLSCREEN',
    'COLON_SCREEN', 'COREM', 'COREW', 'DENTAL', 'MAMMOUSE', 'PAPTEST']

behavior_cols = ['BINGE', 'CSMOKING', 'LPA', 'OBESITY', 'SLEEP']

outcome_cols = [
    'ARTHRITIS', 'BPHIGH', 'CANCER',
    'CASTHMA', 'CHD', 'COPD',
    'DIABETES', 'HIGHCHOL', 'KIDNEY',
    'MHLTH', 'PHLTH', 'STROKE', 'TEETHLOST']

columns_to_drop = [ 'Year', 'StateAbbr', 'DataSource', 'Measure','Data_Value_Unit',
                    'Data_Value_Footnote', 'Data_Value_Type', 'Low_Confidence_Limit',
                    'High_Confidence_Limit', 'Data_Value_Footnote_Symbol',
                    'CategoryID', 'Short_Question_Text']

columns_to_keep = ['StateDesc', 'Category', 'CityName', 'UniqueID', 'GeographicLevel', 'DataValueTypeID',
                   'PopulationCount', 'CityFIPS', 'TractFIPS', 'GeoLocation']

random_state = 100