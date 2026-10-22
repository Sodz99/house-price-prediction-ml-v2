#importing required packages
from sklearn.model_selection import train_test_split
from ML_Pipeline.grid_model import grid_model
from ML_Pipeline.scaler import std_scaler
from ML_Pipeline.utils import read_data
from ML_Pipeline.utils import merge_dataset
from ML_Pipeline.train_model import train_model
from ML_Pipeline.duplicates import remove_duplicates
from ML_Pipeline.label_encoding import encode_categorical_variables
from ML_Pipeline.missing_value import drop_columns_with_max_missing_values, replace_missing_data
from ML_Pipeline.outliers import remove_outliers
from ML_Pipeline.utils import drop_col
from ML_Pipeline.rescale_variables import rescale_variables
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.simplefilter(action='ignore')

# Read the initial datasets
train_2016_v2 = read_data("../input/train_2016_v2.csv")
properties_2016 = read_data("../input/properties_2016.csv")
print("Data Read!") 

# Merge the two datasets
df = merge_dataset(train_2016_v2,properties_2016,join_type='inner',on_param = 'parcelid')

# Remove the total duplicate values present in the dataset
df = remove_duplicates(df, 'parcelid')

# Remove the columns with duplicate values more then 60%
df = drop_columns_with_max_missing_values(df)

# Handling missing values - Replace the rest missing values with mode
mis_var = [var for var in df.columns if df[var].isnull().sum() > 0]
df = replace_missing_data(df, mis_var)

# Capture Elasped Time - Replace the two columns with a new column that contains their difference and then drop those columns
df['yeardifference'] = df['assessmentyear'] - df['yearbuilt']
df.drop(columns=['assessmentyear', 'yearbuilt', 'transactiondate'], axis=1, inplace=True)

# Rescale variables -Some features are incorrectly scaled, and thus are divided with appropriate values to get them back to original scale values
df = rescale_variables(df, ['latitude', 'longitude', 'rawcensustractandblock'], (10 ** 6))
df = rescale_variables(df, ['censustractandblock'], (10 ** 12))
df = rescale_variables(df, [ 'rawcensustractandblock'], (10**6))

# Label encoding - Perform label encoding on categorical variables.
cat_vars = [var for var in df.columns if df[var].dtypes == 'O']
df = encode_categorical_variables(df, cat_vars)

# Outliers - find the outliers through z score
df = remove_outliers(df)

# Drop the unnecessary columns -
# Here,some columns have high multicollinearity present, hence we drop such columns 
# also drop columns like propertycountylandusecode_labels ,parcelid as they serve no useful purpose
df = drop_col(df,['calculatedbathnbr', 'calculatedfinishedsquarefeet', 'structuretaxvaluedollarcnt',
                   'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'fullbathcnt',
                   'censustractandblock', 'propertycountylandusecode_labels','parcelid'])
print("Preprocessing Done!")

# Train-test split - Perform train-test split on the data, 75-25 split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['logerror'],axis=1), df['logerror'], test_size=0.25, random_state = 100)

# Feature Scaling
X_train = std_scaler(X_train)
X_test = std_scaler(X_test)

# Train the models
print("Model Training Starting!")
test = train_model(X_train,X_test,y_train,y_test)
best_model = test[0]
mae_val = test[1]

# Using GridSearchCV for tuning the required model(in this case we are using GridsearchCV on random forest)
# (This can be performed for any model by changing the param_grid accordingly)
param_grid = [{'n_estimators': [400, 500, 600],
               'max_features': [4, 6, 8], 
                'bootstrap': [False]}]

print("Grid Search Starting!")
model_grid_search = grid_model(X_train,y_train,X_test, param_grid,'forest_reg') 

# Dump the model in output folder
joblib.dump(model_grid_search, '../output/final_best_fitted_model.pkl' )
print('Best Model saved as pkl file in '+str('../output/final_best_fitted_model.pkl'))  

# Feature Importance
feature_importances =  model_grid_search.best_estimator_.feature_importances_  
attrs = list(df.select_dtypes(include = ['float64','int64']))
sorted(zip(attrs, feature_importances), reverse=True)






























