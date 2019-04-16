import logging
import sys
import warnings

import numpy as np
import pandas as pd

from constants import *
from get_data import get_data
from util import feature_selection, visual_data_prep, append_existing_data, \
    multi_model_analysis


def initial_setup():
    warnings.filterwarnings("ignore")  # ignore warnings from sklean
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    logger.info("Logger ready")


def main():
    initial_setup()
    print("Accepted mode: \n1. Perform top feature selection Analysis.\n"
          "2. Generate output file for visualization.\n"
          "3. Run multiple models comparision analysis.\n")
    try:
        mode = int(input('Enter the number of your selected mode:\n'))
    except ValueError:
        print("Not a number")

    # Local path to store the raw data
    path = "500_Cities__Local_Data_for_Better_Health__2018_release.csv"
    data, city_pv, tract_pv = get_data(path)
    tract_pv.reset_index(level=0, inplace=True)
    logging.info("Get formatted pivot table data by city and census track from file: " + path)

    if mode == 1:
        print("You have chose to run top feature selection analysis for health care data")
        top_n = int(input('Enter the number of top features you want to select:\n'))

        logging.info("About to run feature selection for city level data")
        feature_selection_df = feature_selection(city_pv, outcome_cols, top_n)
        feature_selection_df.to_csv("city_data.csv")
        logging.info("Country data has been saved to csv")

        logging.info("About to run feature selection for each state")
        states = np.unique(tract_pv['StateDesc'])
        state_feature_selection_df = pd.DataFrame()
        for state in states:
            logging.info("Feature selection for state %s" % state)
            state_data = tract_pv[tract_pv['StateDesc'] == state]
            df = feature_selection(state_data, outcome_cols, top_n)
            df["state"] = state
            state_feature_selection_df = pd.concat([state_feature_selection_df, df])
        state_feature_selection_df.to_csv("state_feature_selection.csv")
        logging.info("States data has been saved to csv")
    elif mode == 2:
        print("You have chose to generate output file for visualization")

        result = pd.DataFrame()
        logging.info("Start data prep in US level")
        prev_data = data[data['GeographicLevel'] == 'US']

        #df = visual_data_prep(city_pv, prev_data, outcome_cols, 5)
        df = visual_data_prep(tract_pv, prev_data, outcome_cols, 5)
        df['Region'] = 'US'
        df['Level'] = 'US'
        df['PreventionCalculationLevel'] = 'National'
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
            df = append_existing_data(result[result.Level == 'US'], prev_data)
            df["Region"] = state
            df['Level'] = 'State'
            df['PreventionCalculationLevel'] = 'National'
            state_df = pd.concat([state_df, df])

            # For each cities with in the state
            cities = np.unique(state_data['CityName'])
            national_calc_df = state_df[state_df.PreventionCalculationLevel == 'National']
            national_calc_df = national_calc_df[national_calc_df.Region == state]
            state_calc_df = state_df[state_df.PreventionCalculationLevel == 'State']
            state_calc_df = state_calc_df[state_calc_df.Region == state]

            city_df = pd.DataFrame()
            for city in cities:
                # Apply national level analysis result for each city
                city_prev_data = prev_data[prev_data.CityName == city]
                df = append_existing_data(national_calc_df, city_prev_data)
                df["Region"] = city
                df['Level'] = 'City'
                city_df = pd.concat([city_df, df])

                # Apply state level analysis result for each city if exists
                if state_calc_df.shape[0] > 0:
                    df = append_existing_data(state_calc_df, city_prev_data)
                    df["Region"] = city
                    df['Level'] = 'City'
                    city_df = pd.concat([city_df, df])
            result = pd.concat([result, city_df])
        result = pd.concat([result, state_df])
        col_order = ['Region','Level','Outcome','Prevention','OutcomePrevalence','PredictionAccuracyScore','PreventionPrevalence','PreventionRank','PreventionCalculationLevel']
        result = result [col_order]
        logging.info("Complete data prep in State level")

        result.to_csv("visual_input_tract_with_city.csv", index=False)
        logging.info("csv file for visualization has been generated")
    elif mode == 3:
        print("You have choose to run multiple models comparision analysis.\n")
        level = input("Enter the level of analysis you want to perform (State/US):\n")
        if level == 'US':
            print("About to analysis on national level")
            multi_model_analysis(city_pv)
        elif level == 'State':
            show_state = input("Do you wish to see all the state names? (Y/N)")
            if show_state == 'Y':
                print( np.unique(tract_pv['StateDesc']) )

            state = input("Enter the name of the state you want to analyze:\n")
            state_data = tract_pv[tract_pv['StateDesc'] == state]
            if state_data.shape[0] == 0:
                print("Bad state name, abort!")
                sys.exit()
            print("About to run model selection for state: ", state, " with ", len(state_data), " census track.")
            multi_model_analysis(state_data)
        else:
            print("Unacceptable input, abort!")


main()
