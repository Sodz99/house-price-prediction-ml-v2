import xgboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from ML_Pipeline.model_evaluation import evaluate_model
from ML_Pipeline.utils import min_val_index
import warnings
warnings.simplefilter(action='ignore')

#Function to build model
def train_model(X_train, X_test, y_train, y_test):
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

    fitted_model=[]
    mae=[]
    
    for model_name in list(model_dict.keys()):   
        model = model_dict[model_name]()
        fitted_model.append(model.fit(X_train, y_train))
        mae.append(evaluate_model(y_test, model.predict(X_test), 'mae'))
            
    min_test = min_val_index(mae)
    min_mae = min_test[0]
    min_mae_index = min_test[1]
    final_model = fitted_model[min_mae_index]
    return (final_model, min_mae)
 
