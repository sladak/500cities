import pandas as pd

from constants import *


def get_data(path):
    data = pd.read_csv(path)

    # drop unecessary columns
    data = data.drop(columns=columns_to_drop)
    # print(data.columns)

    # remove age-adjusted data
    data = data.drop(data[data.DataValueTypeID == 'AgeAdjPrv'].index)

    census_tract_data = data[data['GeographicLevel'] == 'Census Tract']
    city_data = data[data['GeographicLevel'] == 'City']

    tract_pv = census_tract_data.pivot_table(index=['StateDesc', 'CityName', 'UniqueID'], columns='MeasureId',
                                             values='Data_Value',
                                             aggfunc='sum')
    print("Size of census tract data:", len(tract_pv))  # 28004
    tract_pv = tract_pv.fillna(tract_pv.mean())

    city_pv = city_data.pivot_table(index=['StateDesc', 'CityName', 'UniqueID'], columns='MeasureId',
                                    values='Data_Value', aggfunc='sum')
    print("Size of city data:", len(city_pv))  # 500
    city_pv = city_pv.fillna(city_pv.mean())

    city_pv.reset_index(level=0, inplace=True)
    tract_pv.reset_index(level=0, inplace=True)

    return city_pv, tract_pv
