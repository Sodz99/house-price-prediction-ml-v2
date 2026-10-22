# House Price Prediction Project using Machine Learning in Python

To predict the sale prices of the houses and improve the log error i.e.

## Overview
Buying a house that suits his / her choices is every person's desire and it is thus known as their dream house. There are several aspects that one considers while buying a house starting from the budget, the location, the number of rooms available, and many more. But how to find a house that satisfies one’s requirements? This is not a quick and easy task. But no need to worry, homebuyers can nowadays find their dream home with a click of a button. Zillow is a popular estimator for house evaluation available online. It is considered one of the top real estate marketplaces for buying a house in the United States. Zillow’s Zestimate provides the homebuyers to search for a house that satisfies their requirements of location, area, budget, etc. The Zillow Zestimate provides the homebuyers with information on the real worth of the house based on public data. The accuracy of the Zestimate information depends on the location and availability of the data of a specific area. Hence the more data 

## Data
The Zillow Zestimate dataset is a dataset from Kaggle, that is used for making future sales predictions and improving the log error. There are two datasets available –
- train_2016 – This dataset consists of the target variable i.e. the logerror
- properties_2016 – Contents all the features related to the property/home.

There are around 60 attributes in the dataset on basis of which the model can be built. Some of the important features amongst them are as follows:
- train_2016:
1. parcelid - Unique identification for every parcel
2. transactiondate - The date on which the home was sold
3. logerror - The residual between the actual and the predicted sale price of homes. (Target Array)

_Data is not included in this repository._

## Approach
1. Importing the required libraries and reading the dataset.
- Merging of the two datasets
- Understanding the dataset
2. Exploratory Data Analysis (EDA) –
- Data Visualization
3. Feature Engineering
- Duplicate value removal
- Missing value imputation
- Rescaling of incorrectly scaled data
- Standardization
- Encoding of categorical variables
- Generation of new feature wherever required.
- Dropping of redundant feature columns
- Checking for multi-collinearity and removal of highly correlated features
- Check for the outliners and removal of outliers.

4. Model Building
- Performing train test split
- Feature Scaling
- Dropping features if necessary
- Linear Regression Model
- Elastic Net
- Ridge Regression
- Lasso Regressor
- XGBoost Regressor
- Adaboost Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor
- Random Forest Regressor
5.   Model Validation
- Mean Absolute Error
6.   Hypermeter Tuning (GridSearchCV)
- For Random Forest Regressor
7.   Checking for Feature Importance
8.   Creating the final model and making predictions

Modular code overview

Once you unzip the modular_code.zip file you can find the following folders within it.
1. input
2. src
3. output
4. lib
1. Input folder - It contains all the data that we have for analysis. There are two csv files in our case,
- train_2016_v2
- properties_2016

2. src folder - This is the most important folder of the project. This folder contains all the modularized code for all the above steps in a modularized manner. This folder consists of:
- engine.py
- ML_Pipeline The ML_pipeline is a folder that contains all the functions put into different python files which are appropriately named. These python functions are then called inside the engine.py file.
3. Output folder – The output folder contains the best fitted model that we trained for this data. This model can be easily loaded and used for future use and the user need not have to train all the models from the beginning. Note: This model is built over a chunk of data. One can obtain the model for the entire data by running engine.py by taking the entire data to train the models.
4. lib folder - This is a reference folder. It contains the original ipython notebook that we saw in the videos.

## Tech stack
- Language - Python
- Libraries - Scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, xgboost, joblib

## How to run
```bash
pip install -r Modular_Code/Zillow-Prod-Code/House-Price-Original-Codes/requirements.txt
python Modular_Code/Zillow-Prod-Code/House-Price-Original-Codes/src/engine.py
python Modular_Code/src/engine.py
```
Notebooks: [`Zillow_Home_Value_Feature_Engineering.ipynb`](Modular_Code/lib/Zillow_Home_Value_Feature_Engineering.ipynb), [`Zillow_Home_Value_Model_Creation.ipynb`](Modular_Code/lib/Zillow_Home_Value_Model_Creation.ipynb), [`Zillow_Home_Value_Prediction_EDA.ipynb`](Modular_Code/lib/Zillow_Home_Value_Prediction_EDA.ipynb)

## Project structure
```
.gitignore
Modular_Code/
  Zillow-Prod-Code/
    .idea/
    House-Price-Original-Codes/
    README.md
  lib/
    Zillow_Home_Value_Feature_Engineering.ipynb
    Zillow_Home_Value_Model_Creation.ipynb
    Zillow_Home_Value_Prediction_EDA.ipynb
  requirements.txt
  src/
    ML_Pipeline/
    engine.py
README.md
```

## Credits
Based on a ProjectPro guided project.
