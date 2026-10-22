import pandas as pd
import xgboost
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from math import sqrt
import warnings
warnings.simplefilter(action='ignore')


def train_and_save_model(df):
    X_train, X_test, y_train, y_test = do_train_test_split(df)
    X_train, X_test = perform_feature_scaling(X_train, X_test)
    model_path = '../output/random_forest_reg.pkl'
    print('##### Performing Model Training #####')
    train_random_forest_reg(X_train, y_train, model_path)
    return model_path, X_test, y_test


def do_train_test_split(df):
    print('##### Performing Train Test Split #####')
    X = df.drop('logerror', axis=1)
    y = df['logerror']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
    return X_train, X_test, y_train, y_test


def perform_feature_scaling(X_train, X_test):
    print('##### Performing Feature Scaling #####')
    train_vars = [var for var in X_train.columns if var not in ['parcelid', 'logerror']]
    scaler = MinMaxScaler()
    scaler.fit(X_train[train_vars])
    X_train[train_vars] = scaler.transform(X_train[train_vars])
    X_test[train_vars] = scaler.transform(X_test[train_vars])
    return X_train, X_test


def train_random_forest_reg(X_train, y_train, model_path):
    param_grid = [
        {'n_estimators': [300, 400, 500, 600], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [300, 400, 500, 600], 'max_features': [2, 4, 6, 8]}
    ]
    forest_regressor = RandomForestRegressor()

    forest_reg = hyperparameter_optimization(forest_regressor, param_grid, X_train, y_train)
    forest_reg.fit(X_train, y_train)
    pickle.dump(forest_reg, open(model_path, 'wb'))
    return None


def hyperparameter_optimization(regressor, param_grid, X_train, y_train):
    grid_search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=10)
    grid_search.fit(X_train, y_train)
    print('##### Performing Hyper Parameter Optimization #####')
    print('Best Parameters : ', grid_search.best_params_)

    feature_importance = grid_search.best_estimator_.feature_importances_
    attrs = list(X_train.select_dtypes(include=['float64', 'int64']))
    print('Displaying Feature Importance : ')
    print(sorted(zip(attrs, feature_importance), reverse=True))

    return grid_search.best_estimator_


def model_prediction(model_path, X_test, y_test):
    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(X_test)
    print('Mean Absolute Error : {}'.format(mean_absolute_error(y_test, predictions)))
    print('Mean Squared Error : {}'.format(mean_squared_error(y_test, predictions)))
    print('Root Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, predictions))))

    prediction_df = pd.DataFrame({'parcelid': X_test.parcelid, 'logerror': predictions})
    path = '../output/random_forest_predictions.csv'
    prediction_df.to_csv(path, index=False)
    return None


def run_all_regression_models(df):
    X_train, X_test, y_train, y_test = do_train_test_split(df)
    X_train, X_test = perform_feature_scaling(X_train, X_test)

    print('##### Training & evaluating Linear Regression Model #####')
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    linear_reg_pred = linear_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, linear_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, linear_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, linear_reg_pred))))

    print('##### Training & evaluating Elastic Net Model #####')
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    elastic_net_pred = elastic_net.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, elastic_net_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, elastic_net_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, elastic_net_pred))))

    print('##### Training & evaluating Ridge Regression Model #####')
    ridge_reg = Ridge(alpha=1, solver='cholesky')
    ridge_reg.fit(X_train, y_train)
    ridge_reg_pred = ridge_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, ridge_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, ridge_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, ridge_reg_pred))))

    print('##### Training & evaluating Lasso Regression Model #####')
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X_train, y_train)
    lasso_reg_pred = lasso_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, lasso_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, lasso_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, lasso_reg_pred))))

    print('##### Training & evaluating XG Boost Regression Model #####')
    xgb_reg = xgboost.XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    xgb_reg_pred = xgb_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, xgb_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, xgb_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, xgb_reg_pred))))

    print('##### Training & evaluating Ada Boost Regression Model #####')
    adaboost_reg = AdaBoostRegressor()
    adaboost_reg.fit(X_train, y_train)
    adaboost_reg_pred = adaboost_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, adaboost_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, adaboost_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, adaboost_reg_pred))))

    print('##### Training & evaluating Gradient Boosting Regression Model #####')
    gb_reg = GradientBoostingRegressor()
    gb_reg.fit(X_train, y_train)
    gb_reg_pred = gb_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, gb_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, gb_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, gb_reg_pred))))

    print('##### Training & evaluating Decision Tree Regression Model #####')
    tree_reg = DecisionTreeRegressor(max_depth=5)
    tree_reg.fit(X_train, y_train)
    tree_reg_pred = tree_reg.predict(X_test)
    print('\nMean Absolute Error : {}'.format(mean_absolute_error(y_test, tree_reg_pred)))
    print('\nMean Squared Error : {}'.format(mean_squared_error(y_test, tree_reg_pred)))
    print('\nRoot Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, tree_reg_pred))))

    return None



