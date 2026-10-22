import numpy as np
from scipy import stats

# Function to remove the outliners with z score,  thershold 3 

def remove_outliers(df):

    z = np.abs(stats.zscore(df))
    df_new = df[(z < 3).all(axis=1)]
    return df_new
