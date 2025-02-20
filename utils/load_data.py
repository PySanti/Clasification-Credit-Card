import pandas as pd
from pandas.core.common import random_state
from preprocess.features_selection import features_selection
from sklearn.model_selection import train_test_split

def load_data(path : str, target_value : str, use_splitted_dataset=False):

    relevant_features = ['V17','V14','V12','V16','V10','V11','V9','V18','V4','V7','V3','V26','V21','V1','V8','V5','V19','V2','Time','V20','V6','Amount','V13','V15']
    df = pd.read_csv(path)
    df = df[relevant_features+[target_value]]

    if use_splitted_dataset:
        split_df = df.copy()
        df_train, df_test = train_test_split(split_df, test_size=1-use_splitted_dataset, shuffle=True, random_state=42, stratify=(df[target_value]))
        df = df_train


    return [df.drop(target_value, axis=1), df[target_value]]
