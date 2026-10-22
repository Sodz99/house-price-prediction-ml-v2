import numpy as np

# Function to drop columns with missing values more then 60 %

def drop_columns_with_max_missing_values(df):

    mis_var = [var for var in df.columns if df[var].isnull().sum() > 0]
    df[mis_var].isnull().sum()
    limit = np.abs((df.shape[0] * 0.6))
    var_to_be_dropped = [var for var in mis_var if df[var].isnull().sum() > limit]
    df.drop(columns=var_to_be_dropped, axis=1, inplace=True)
 
    return df

# Function to deal with missing values in numerical variables. Replacing with mode
def replace_missing_data(df, mis_var):

    for var in mis_var:
        df[var] = df[var].fillna(df[var].mode()[0])
    return df
