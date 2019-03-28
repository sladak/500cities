import logging
import warnings

import statsmodels.api as sm

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR

from get_data import get_data
from constants import *
from util import analyze_state_level_data


def setup_log():
    global logger
    # output_dir = "/tmp/somefile"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(output_dir, "w")
    # handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    logger.info("Logger ready")


warnings.filterwarnings("ignore")  # ignore warnings from sklean
setup_log()

# Change this to your local path
path = "500_Cities__Local_Data_for_Better_Health__2018_release.csv"
city_pv, tract_pv = get_data(path)
logging.info("Get formatted pivot table data by city and census track from file: " + path)

# TODO: For states in util.py, write a loop to analyze data for each outcome
clean_data = city_pv
x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]

for outcome in range(0, len(outcome_cols)):
    outcome_col = outcome_cols[outcome]
    print ()
    print ("**********  OUTCOME", outcome, ":", outcome_col," **********")
    y_data = clean_data.loc[:, outcome_col]

#    random_state = 100  # to make the result reproducible ALREADY IMPORTED FROM CONSTANTS FILE
    # split data to training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=random_state)

    # TODO: seperate them into a function
    # Linear Regression
    reg = LinearRegression().fit(x_train, y_train)
    print("lm coef:", reg.coef_)
    print("lm train score:", round(reg.score(x_train, y_train), 4))
    print("lm test score:", round(reg.score(x_test, y_test), 4))
    
#    X = sm.add_constant(x_train)
#    lm = sm.OLS(y_train, X).fit()
#    print(lm.summary())
    
    # Ridge Regression
    reg = Ridge().fit(x_train, y_train)
    print("Ridge coef:", reg.coef_)
    print("Ridge train score:", reg.score(x_train, y_train))
    print("Ridge test score:", reg.score(x_test, y_test))
    
    # Lasso Regression
    reg = Lasso().fit(x_train, y_train)
    print("Lasso coef:", reg.coef_)
    print("Lasso train score:", reg.score(x_train, y_train))
    print("Lasso test score:", reg.score(x_test, y_test))
    
    # SVM - regression with Hyper-parameter Tuning
    svr = SVR()
    # clf = svr.fit(x_train, y_train)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 1, 10]}
    clf = GridSearchCV(svr, parameters, cv=10).fit(x_train, y_train)
    best_param = clf.best_params_
    print("svm best params:", clf.best_params_)
    print("svm best score:", clf.best_score_)
    print("svm best set train accuracy:", round(r2_score(y_train, clf.predict(x_train)), 2))
    print("svm best set test accuracy:", round(r2_score(y_test, clf.predict(x_test)), 2))
    # print(clf.cv_results_)
    
    # feature selection - recursive feature elimination
    estimator = SVR(kernel=best_param['kernel'], C=best_param['C'])
    selector = RFE(estimator, 5, step=1)  # top 5 features
    selector = selector.fit(x_train, y_train)
    # print(x_test.columns)
    # print(selector.support_)
    # print(selector.ranking_)
    print("top features:", x_test.columns[selector.support_])


analyze_state_level_data(tract_pv)
# TODO: Random Forest Regression? (for non-Linear boundaries)

# TODO: model comparison - select the best model based on score?


# TODO: Generate results in a proper format
# GeoLevel                      | Best Model | Top 5 params    | Test Score | ....
# Country (By city)             | svm        | [a,b,c,d,e]    | 0.8542      |....
# City/State? (By census tract) | Lasso      | [d,e,f,g,h]   |  0.8624      |....
