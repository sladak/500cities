import logging
import warnings

import numpy as np
import pandas as pd

from constants import *
from get_data import get_data
from util import feature_selection, visual_data_prep, append_existing_data


def initial_setup():
    warnings.filterwarnings("ignore")  # ignore warnings from sklean
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    logger.info("Logger ready")


def main(mode='Visual'):
    initial_setup()
    print("Accepted mode: \n1. Perform Analysis.\n2. Generate output file for visualization")
    try:
        mode = int(input('Enter the number of your selected mode:'))
    except ValueError:
        print("Not a number")

    # Local path to store the raw data
    path = "500_Cities__Local_Data_for_Better_Health__2018_release.csv"
    data, city_pv, tract_pv = get_data(path)
    tract_pv.reset_index(level=0, inplace=True)
    logging.info("Get formatted pivot table data by city and census track from file: " + path)

    if mode == 1:
        print("You have chose to run analysis for health care data")

        logging.info("About to run feature selection for city level data")
        feature_selection_df = feature_selection(city_pv, outcome_cols, 5)
        feature_selection_df.to_csv("city_data.csv")
        logging.info("Country data has been saved to csv")

        logging.info("About to run feature selection for each state")
        states = np.unique(tract_pv['StateDesc'])
        state_feature_selection_df = pd.DataFrame()
        for state in states:
            logging.info("Feature selection for state %s" % state)
            state_data = tract_pv[tract_pv['StateDesc'] == state]
            df = feature_selection(state_data, outcome_cols, 5)
            df["state"] = state
            state_feature_selection_df = pd.concat([state_feature_selection_df, df])
        state_feature_selection_df.to_csv("state_feature_selection.csv")
        logging.info("States data has been saved to csv")
    elif mode == 2:
        print("You have chose to generate output file for visualization")

        result = pd.DataFrame()
        logging.info("Start data prep in US level")
        prev_data = data[data['GeographicLevel'] == 'US']

        df = visual_data_prep(tract_pv, prev_data, outcome_cols, 5)
        df['Region'] = 'US'
        df['Level'] = 'US'
        result = pd.concat([result, df])
        logging.info("Complete data prep in US level")

        logging.info("Start data prep in State level")
        states = np.unique(tract_pv['StateDesc'])
        state_df = pd.DataFrame()
        for state in states:
            state_data = tract_pv[tract_pv['StateDesc'] == state]
            logging.info("Data prep for state %s with %s census tracks" % (state, len(state_data)))
            prev_data = data[data['GeographicLevel'] == 'City']
            prev_data = prev_data[prev_data['StateDesc'] == state]

            # for state has more than 100 data points, we conduct another analysis on state level
            if len(state_data) > 100:
                df = visual_data_prep(state_data, prev_data, outcome_cols, 5)
                df['PreventionCalculationLevel'] = 'State'
                df["Region"] = state
                df['Level'] = 'State'
                state_df = pd.concat([state_df, df])

            # Append national level data for each state with state prevalence information
            df = append_existing_data(result[result.Level == 'US'], prev_data, state )

            # For each cities with in the state
            cities = np.unique(state_data['CityName'])
            us_df = result[result.Level == 'US']

            city_df = pd.DataFrame()
            for city in cities:
                temp = pd.DataFrame()
                for outcome_col in outcome_cols:
                    row = prev_data[prev_data.MeasureId == outcome_col]
                    row = row[row.CityName == city]
                    df2 = us_df[us_df['Outcome'] == outcome_col]
                    df2['OutcomePrevalence'] = row.Data_Value.iloc[0]
                    prevention = df2['Prevention'].iloc[:]
                    p_list = []
                    for p in prevention:
                        prevention_row = prev_data[prev_data.MeasureId == p]
                        prevention_row = prevention_row[prevention_row.CityName == city]
                        prevention_prevalence = np.sum(
                            prevention_row.Data_Value * prevention_row.PopulationCount) / np.sum(
                            prevention_row.PopulationCount)
                        p_list.append(prevention_prevalence)
                    df2['PreventionPrevalence'] = p_list
                    temp = pd.concat([temp, df2])
                df = temp
                df["Region"] = city
                df['Level'] = 'City'
                city_df = pd.concat([city_df, df])
            result = pd.concat([result, city_df])
        result = pd.concat([result, state_df])

        logging.info("Complete data prep in State level")

        result.to_csv("visual_input_tract_with_city.csv")
        logging.info("csv file for visualization has been generated")


# analyze_state_level_data can be used to run different models for all the states
# analyze_state_level_data(tract_pv)


# Below code is the initial version to run different models for US given city level data
# TODO: make it to an util function
# clean_data = city_pv
# x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]
#
# i = 0
# for outcome_col in outcome_cols:
#     print("-----------------------------------")
#     print("#", i)
#
#     print("Health Outcome to analyze: ", outcome_col)
#     i += 1
#     y_data = clean_data.loc[:, outcome_col]
#
#     # split data to training and testing set
#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=random_state)
#
#     # TODO: seperate them into a function
#     # Linear Regression
#     reg = LinearRegression().fit(x_train, y_train)
#     # print(reg.intercept_)
#     # print(reg.coef_)
#     print(reg)
#     print("lm train score:", round(reg.score(x_train, y_train), 4))
#     print("lm test score:", round(reg.score(x_test, y_test), 4))
#
#     X = sm.add_constant(x_train)
#     lm = sm.OLS(y_train, X).fit()
#     print(lm.summary())
#
#     # Ridge Regression
#     reg = Ridge().fit(x_train, y_train)
#     print(reg.intercept_)
#     print(reg.coef_)
#     print(reg)
#     print("Ridge train score:", reg.score(x_train, y_train))
#     print("Ridge test score:", reg.score(x_test, y_test))
#
#     # Lasso Regression
#     reg = Lasso().fit(x_train, y_train)
#     print(reg.intercept_)
#     print(reg.coef_)
#     print(reg)
#     print("Lasso train score:", reg.score(x_train, y_train))
#     print("Lasso test score:", reg.score(x_test, y_test))
#
#     # SVM - regression with Hyper-parameter Tuning
#     svr = SVR()
#     # clf = svr.fit(x_train, y_train)
#     parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 1, 10]}
#     clf = GridSearchCV(svr, parameters, cv=10).fit(x_train, y_train)
#     best_param = clf.best_params_
#     print("svm best params:", clf.best_params_)
#     print("svm best score:", clf.best_score_)
#     print("svm best set train accuracy:", round(r2_score(y_train, clf.predict(x_train)), 2))
#     print("svm best set test accuracy:", round(r2_score(y_test, clf.predict(x_test)), 2))
#     # print(clf.cv_results_)
#
#     # feature selection - recursive feature elimination
#     estimator = SVR(kernel=best_param['kernel'], C=best_param['C'])
#     selector = RFE(estimator, 5, step=1)  # top 5 features
#     selector = selector.fit(x_train, y_train)
#     # print(x_test.columns)
#     # print(selector.support_)
#     # print(selector.ranking_)
#     print("top features:", x_test.columns[selector.support_])

# TODO: Random Forest Regression?

# TODO: model comparison - select the best model based on score?
