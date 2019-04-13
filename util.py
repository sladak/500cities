import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from constants import *


def analyze_state_level_data(tract_pv):
    states = np.unique(tract_pv['StateDesc'])
    for state in states:
        state_data = tract_pv[tract_pv['StateDesc'] == state]
        if len(state_data) > 100:
            print("Run model selection for state: ", state, " with ", len(state_data), " census track.")
            # split data to training and testing set
            clean_data = state_data
            x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]
            outcome_col = outcome_cols[0]
            y_data = clean_data.loc[:, outcome_col]
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7,
                                                                random_state=random_state)

            reg = LinearRegression().fit(x_train, y_train)
            print("lm coef:", reg.coef_)
            print("lm train score:", round(reg.score(x_train, y_train), 4))
            print("lm test score:", round(reg.score(x_test, y_test), 4))

            # Lasso Regression
            reg = Lasso().fit(x_train, y_train)
            print("Lasso coef:", reg.coef_)
            print("Lasso train score:", round(reg.score(x_train, y_train), 4))
            print("Lasso test score:", round(reg.score(x_test, y_test), 4))

            # SVM - regression with Hyper-parameter Tuning
            svr = SVR()
            parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 1, 10]}
            clf = GridSearchCV(svr, parameters, cv=10).fit(x_train, y_train)
            best_param = clf.best_params_
            print("svm best set train accuracy:", round(r2_score(y_train, clf.predict(x_train)), 4))
            print("svm best set test accuracy:", round(r2_score(y_test, clf.predict(x_test)), 4))
            # feature selection - recursive feature elimination
            estimator = SVR(kernel=best_param['kernel'], C=best_param['C'])
            selector = RFE(estimator, 5, step=1)  # top 5 features
            selector = selector.fit(x_train, y_train)
            print("top features:", x_test.columns[selector.support_])


# Top n Feature selection on given data for the specified columns
def feature_selection(clean_data, cols, n=5):
    train_score = []
    test_score = []
    top_feature = []
    for outcome_col in cols:
        x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]
        y_data = clean_data.loc[:, outcome_col]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_data)

        x_train_std, x_test_std, y_train, y_test = train_test_split(x_scaled, y_data, train_size=0.7,
                                                                    random_state=random_state)

        param_grid = {'kernel': ['linear'], 'C': [0.01, 1, 10]}
        clf = GridSearchCV(SVR(), param_grid, cv=5)
        clf = clf.fit(x_train_std, y_train)

        train_scr = r2_score(y_train, clf.predict(x_train_std))
        test_scr = r2_score(y_test, clf.predict(x_test_std))

        train_score.append(train_scr)
        test_score.append(test_scr)

        selector = RFE(clf.best_estimator_, 1, step=1)
        selector = selector.fit(x_train_std, y_train)
        top_feature.append(sorted(zip(selector.ranking_, x_data.columns.tolist()), key=lambda x: x[0])[:n])
    d = {'health_outcomes': cols, 'train_score': train_score, 'test_score': test_score, 'top_feature': top_feature}
    return pd.DataFrame(data=d)


def visual_data_prep(clean_data, prev_data, cols, n=5):
    df = pd.DataFrame()
    for outcome_col in cols:
        x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]
        y_data = clean_data.loc[:, outcome_col]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_data)

        x_train_std, x_test_std, y_train, y_test = train_test_split(x_scaled, y_data, train_size=0.7,
                                                                    random_state=random_state)

        param_grid = {'kernel': ['linear'], 'C': [0.01, 1, 10]}
        clf = GridSearchCV(SVR(), param_grid, cv=5)
        clf = clf.fit(x_train_std, y_train)

        train_scr = r2_score(y_train, clf.predict(x_train_std))
        test_scr = r2_score(y_test, clf.predict(x_test_std))

        selector = RFE(clf.best_estimator_, 1, step=1)
        selector = selector.fit(x_train_std, y_train)
        arr_tup = sorted(zip(selector.ranking_, x_data.columns.tolist()), key=lambda x: x[0])[:n]

        prevention = [x[1] for x in arr_tup]
        prevention_rank = [x[0] for x in arr_tup]
        d = {'Prevention': prevention,'PreventionRank':prevention_rank}
        temp = pd.DataFrame(data=d)

        temp['Outcome'] = outcome_col
        temp['PredictionAccuracyScore'] = train_scr

        row = prev_data[prev_data.MeasureId == outcome_col]

        temp['OutcomePrevalence'] = row.Data_Value.iloc[0]
        temp['PreventionPrevalence'] = [prev_data[prev_data.MeasureId == p].Data_Value.iloc[0] for p in prevention]
        df = pd.concat([temp, df])

    return df


def append_existing_data(df, prev_data, region):
    temp = pd.DataFrame()
    for outcome_col in outcome_cols:
        outcome_row = prev_data[prev_data.MeasureId == outcome_col]

        df2 = df[df['Outcome'] == outcome_col]
        # Weighted average prevalence for the outcome
        df2['OutcomePrevalence'] = np.sum(outcome_row.Data_Value * outcome_row.PopulationCount) / np.sum(
            outcome_row.PopulationCount)
        prevention = df2['Prevention'].iloc[:]

        p_list = []
        for p in prevention:
            prevention_row = prev_data[prev_data.MeasureId == p]
            # Weighted average prevalence for the prevention
            prevention_prevalence = np.sum(
                prevention_row.Data_Value * prevention_row.PopulationCount) / np.sum(
                prevention_row.PopulationCount)
            p_list.append(prevention_prevalence)

        df2['Prevention Prevalence'] = p_list
        temp = pd.concat([temp, df2])
    df = temp
    df["Region"] = region
    df['Level'] = 'State'
    df['PreventionCalculationLevel'] = 'National'

# Region	Level	Outcome	Prevention	Outcome Prevelance	Prevention Prevelence	Prevention Rank	Prevention Score
# US	US	Arthritis	Health Insurance	25.4	11.6	1	0.509765339
