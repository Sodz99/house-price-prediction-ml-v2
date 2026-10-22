import numpy as np
import pandas as pd


def merge_dataset(df1, df2, join_type, on_param):
    final_df = df1.copy()
    final_df = final_df.merge(df2, how=join_type, on=on_param)
    return final_df


def save_dataset(df, storage_type):
    if storage_type == 'input':
        df.to_csv('../input/zillow_initial_dataset.csv', index=False)
    elif storage_type == 'output':
        df.to_csv('../output/model_predictions.csv', index=False)
    else:
        df.to_csv('../output/final_zillow_dataset.csv', index=False)
