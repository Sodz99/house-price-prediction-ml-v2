import pandas as pd
from ML_Pipeline import utils
from ML_Pipeline import feature_engineering
from ML_Pipeline import train_model
import warnings
warnings.simplefilter(action='ignore')

try:
    # Load Datasets and join the tables with respect to correct identifiers:
    print('##### Loading Dataset ######')
    features_df = pd.read_csv("../input/properties_2016.csv")
    target_df = pd.read_csv("../input/train_2016.csv")

    initial_dataset = utils.merge_dataset(features_df, target_df, 'inner', 'parcelid')
    print("Shape of Dataset: ", initial_dataset.shape)

    utils.save_dataset(initial_dataset, 'input')

    # Perform Feature Engineering:
    print('##### Performing Feature Engineering #####')
    feature_engineering.perform_feature_engineering(initial_dataset)

    # Train model and save it for future:
    print('##### Load feature engineered dataset fro model training #####')
    final_df = pd.read_csv('../output/final_zillow_data.csv')
    model_path, test_df, test_labels = train_model.train_and_save_model(final_df)

    # Load model and do predictions:
    train_model.model_prediction(model_path, test_df)

except Exception as e:
    print('!! Exception Details: !!\n', e.__class__)
    print('Please debug for further details')



