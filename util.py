import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

from constants import *
import statsmodels.api as sm


def analyze_state_level_data(tract_pv):
    states = np.unique(tract_pv['StateDesc'])
    for state in states:
        state_data = tract_pv[tract_pv['StateDesc'] == state]
        if len(state_data)>400:
            print("Run model selection for state: ", state," with ", len(state_data), " census track.")
            # split data to training and testing set
            clean_data = state_data
            x_data = clean_data.loc[:, clean_data.columns.isin(prevention_cols + behavior_cols)]
            outcome_col = outcome_cols[0]
            y_data = clean_data.loc[:, outcome_col]
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=random_state)

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



