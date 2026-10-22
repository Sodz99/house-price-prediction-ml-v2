import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import utils


def perform_feature_engineering(df):
    print('Dataset Information: \n', df.info())
    print('\nDataset Columns: \n', df.columns)
    print('\nDataset feature value ranges: \n', df.describe())

    new_df = remove_duplicates(df, 'parcelid')

    new_df = drop_columns_with_max_missing_values(df)

    new_df = capture_elapsed_time(new_df)

    new_df = rescale_variables(df, ['latitude', 'longitude', 'rawcensustractandblock'], (10 ** 6))
    new_df = rescale_variables(df, ['censustractandblock'], (10 ** 12))

    mis_var = [var for var in new_df.columns if new_df[var].isnull().sum() > 0]
    new_df = replace_missing_data(new_df, mis_var)

    cat_vars = [var for var in new_df.columns if new_df[var].dtypes == 'O']
    new_df = encode_categorical_variables(new_df, cat_vars)

    new_df = remove_outliers(new_df)

    print('##### Removing Multi-Colinearity #####')
    new_df.drop(
        columns=['calculatedbathnbr', 'calculatedfinishedsquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                 'landtaxvaluedollarcnt', 'fullbathcnt'], axis=1, inplace=True)

    print('##### Removing redundant columns #####')
    new_df.drop(columns=['censustractandblock', 'propertycountylandusecode_labels'], axis=1, inplace=True)

    utils.save_dataset(new_df, 'preprocessing')
    return None


def drop_columns_with_max_missing_values(df):
    mis_var = [var for var in df.columns if df[var].isnull().sum() > 0]
    df[mis_var].isnull().sum()

    limit = np.abs((df.shape[0] * 0.6))
    var_to_be_dropped = [var for var in mis_var if df[var].isnull().sum() > limit]
    print('Columns with more than 60% missing values: \n', var_to_be_dropped)
    print('##### Dropping columns with more than 60% missing values #####')

    df.drop(columns=var_to_be_dropped, axis=1, inplace=True)
    print('Remaining columns: \n', df.columns)
    return df


def encode_categorical_variables(df, cat_vars):
    print('##### Performing Label Encoding on Categorical Variables #####')
    print('Categorical variables: ', cat_vars)
    for i in range(len(cat_vars)):
        var = cat_vars[i]
        var_le = LabelEncoder()
        var_labels = var_le.fit_transform(df[var])
        var_mappings = {index: label for index, label in enumerate(var_le.classes_)}
        df[(var + '_labels')] = var_labels
        df.drop(columns=var, axis=1, inplace=True)
    return df


def remove_outliers(df):
    print('##### Removing outliers from dataset with Z score values greater than 3 #####')
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]
    return df


def replace_missing_data(df, mis_vars):
    print('##### Replacing missing values with mode of features #####')
    for var in mis_vars:
        df[var] = df[var].fillna(df[var].mode()[0])
    return df


def capture_elapsed_time(df):
    print('##### Capturing time difference as a new feature in temporal variables #####')
    year_var = [var for var in df.columns if 'Yr' in var or 'year' in var and var != 'taxdelinquencyyear']
    df['yeardifference'] = df[year_var[1]] - df[year_var[0]]
    df.drop(columns=year_var, axis=1, inplace=True)
    return df


def remove_duplicates(df, value):
    duplicate = df[df.duplicated(value)]
    df.drop_duplicates(subset=value, keep='first', inplace=True)
    return df


def rescale_variables(df, column_list, value):
    for var in column_list:
        df[var] = (df[var]) / value
    return df
