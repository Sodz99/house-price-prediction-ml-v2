from sklearn.model_selection import GridSearchCV
import xgboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import warnings
warnings.simplefilter(action='ignore')

def grid_model(X_train,y_train,X_test,param_grid,best_model_name):
  model_dict = {
        'linear_reg': LinearRegression,
        'elastic_net': ElasticNet,
        'ridge_reg' : Ridge,
        'lasso_reg' : Lasso,
        'xgb_reg' : xgboost.XGBRegressor,
        'adaboost_reg' : AdaBoostRegressor,
        'gb_reg' : GradientBoostingRegressor,
        'tree_reg' : DecisionTreeRegressor,
        'forest_reg' : RandomForestRegressor
    }
  model = model_dict[best_model_name]()
  model.fit(X_train, y_train)

  grid_search = GridSearchCV(model, param_grid,cv=3, n_jobs = -1)
  grid_search.fit(X_train,y_train)
  final_predictor = grid_search.best_estimator_
  final_predictor.fit(X_train, y_train)
  final_pred = final_predictor.predict(X_test)
  return grid_search
