"""
Try various hyperparameters.
"""

# import libraries

import joblib
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# define tuned parameters

# trial 3
tuned_params = {
    'num_leaves': 357, 
    'min_data_in_leaf': 239, 
    'bagging_fraction': 0.8981245960747626, 
    'bagging_freq': 42, 
    'feature_fraction': 0.6757366405684708, 
    'learning_rate': 0.002643965632778826, 
    'lambda_l1': 0.00020908461399848322, 
    'lambda_l2': 2.282598591643265e-06, 
    'min_split_gain': 0.00539175228082824
}

model_name = 'lgbm_trial_3'
output_name = 'prediction_trial_3'

# define paths

model_dir_path = 'models/lgbm/'
output_dir_path = 'output/lgbm/'

train_transaction_data_path = 'data/train_transaction.csv'
train_identity_data_path = 'data/train_identity.csv'
test_transaction_data_path = 'data/test_transaction.csv'
test_identity_data_path = 'data/test_identity.csv'


# define utility function to reduce memory usage

def reduce_mem_usage(df, verbose=True):
    """
    Reduce dataframe size

    params:
    - df: dataframe to reduce the size of

    return:
    - dataframe of reduced size
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'float128']
    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)
                elif c_min > np.finfo(np.float128).min and c_max < np.finfo(np.float128).max:
                    df[col] = df[col].astype(np.float128)
                    
    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: 
        print(
            'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        ))

    return df


# list down useless features (known from feature selection)

useless_features = [
    'TransactionID',  # not really a feature
    'dist2',  # transaction features
    'C3',  # C features
    'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14',  # D features
    'M1',  # M features
    'id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23',  # id features
    'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_35',  # id features
    'V6', 'V8', 'V9', 'V10', 'V11', 'V14', 'V15', 'V16',  # V features
    'V18', 'V21', 'V22', 'V27', 'V28', 'V31', 'V32',  # V features
    'V41', 'V42', 'V46', 'V50', 'V51', 'V59', 'V65',  # V features
    'V68', 'V71', 'V72', 'V79', 'V80', 'V84', 'V85',  # V features
    'V88', 'V89', 'V92', 'V93', 'V95', 'V98', 'V101',  # V features
    'V104', 'V106', 'V107', 'V108', 'V109', 'V110',  # V features
    'V111', 'V112', 'V113', 'V114', 'V116', 'V117',  # V features
    'V118', 'V119', 'V120', 'V121', 'V122', 'V123',  # V features 
    'V125', 'V138', 'V141', 'V142', 'V144', 'V146',  # V features 
    'V147', 'V148', 'V151', 'V153', 'V154', 'V155',  # V features 
    'V157', 'V158', 'V159', 'V161', 'V163', 'V164',  # V features 
    'V166', 'V172', 'V173', 'V174', 'V175', 'V176',  # V features 
    'V177', 'V178', 'V179', 'V180', 'V181', 'V182',  # V features  
    'V183', 'V184', 'V185', 'V186', 'V190', 'V191',  # V features  
    'V192', 'V193', 'V194', 'V195', 'V196', 'V197',  # V features  
    'V198', 'V199', 'V214', 'V216', 'V220', 'V225',  # V features 
    'V226', 'V227', 'V230', 'V233', 'V235', 'V236',  # V features  
    'V237', 'V238', 'V239', 'V240', 'V241', 'V242',  # V features 
    'V244', 'V246', 'V247', 'V248', 'V249', 'V250',  # V features 
    'V252', 'V254', 'V255', 'V269', 'V276', 'V297',  # V features 
    'V300', 'V302', 'V304', 'V305', 'V325', 'V327',  # V features  
    'V328', 'V329', 'V330', 'V334', 'V335', 'V336',  # V features 
    'V337', 'V338', 'V339',  # V features 
]


# define function to disregard OS versions

def ignore_os_version(df, verbose: bool=True):
    """
    params:
    - df (DataFrame): has id_30 as one of its columns
    - verbose (bool): prints information if True

    return: dataframe, after os versions have been ignored
    """
    os_list = [
        'Android',
        'iOS',
        'Mac OS X',
        'Windows',
    ]

    for index, operating_system in df.id_30.iteritems():
        new_os = 'other'

        if isinstance(operating_system, str):
            for known_os in os_list:
                if known_os in operating_system:
                    new_os = known_os
                    break

        df.at[index, 'id_30'] = new_os

    if verbose:
        print('operating systems:', df.id_30.unique())

    return df


# define function to disregard browser versions

def ignore_browser_version(df, verbose: bool=True):
    """
    params:
    - df (DataFrame): has id_31 as one of its columns
    - verbose (bool): prints information if True

    return: dataframe, after browser versions have been ignored
    """
    browser_list = [
        'aol',
        'chrome',
        'chromium',
        'comodo',
        'cyberfox',
        'edge',
        'firefox',
        'icedragon',
        'ie',
        'iron',
        'maxthon',
        'opera',
        'palemoon',
        'puffin',
        'safari',
        'samsung',
        'seamonkey',
        'silk',
        'waterfox',
    ]

    for index, browser in df.id_31.iteritems():
        new_browser = 'other'

        if isinstance(browser, str):
            for known_browser in browser_list:
                if known_browser in browser:
                    new_browser = known_browser
                    break

        df.at[index, 'id_31'] = new_browser

    if verbose:
        print('browsers:', df.id_31.unique())

    return df


# define function for preprocessing data

def preprocess(df, verbose: bool=True):
    """
    Does the following preprocessing steps:
    - disregard os versions
    - disregard browser versions
    - drop useless features
    - convert object columns to string columns
    - imputation (for numbers, fill with interquartile mean)
    - do label encoding for non-numeric values
    - reduce memory usage again

    params:   
    - df (DataFrame): dataframe to preprocess (has columns id_30 and id_31)
    - verbose (bool): prints information if True

    return: dataframe, preprocessing is complete
    """
    df = df.drop(useless_features, axis=1)
    df = ignore_os_version(df, verbose)
    df = ignore_browser_version(df, verbose)

    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column]= df[column].astype(str)
            df[column] = le.fit_transform(df[column])
        else:
            df[column] = df[column].fillna(df[column].quantile().mean())

    df = reduce_mem_usage(df, verbose)

    return df


# define function to load and preprocess training data

def load_training_data():
    transaction_dataframe = pd.read_csv(train_transaction_data_path)
    transaction_dataframe = reduce_mem_usage(transaction_dataframe)

    identity_dataframe = pd.read_csv(train_identity_data_path)
    identity_dataframe = reduce_mem_usage(identity_dataframe)

    dataframe = transaction_dataframe.merge(identity_dataframe, how='outer')

    del transaction_dataframe
    del identity_dataframe

    print(f'number of rows in training data: {len(dataframe)}')
    dataframe = preprocess(dataframe)

    features_dataframe = dataframe.drop('isFraud', axis=1)
    is_fraud_data = dataframe['isFraud']

    del dataframe

    train_features, val_features, train_target, val_target = train_test_split(
        features_dataframe, 
        is_fraud_data, 
        test_size=0.2,
    )

    train_data = lgb.Dataset(train_features, train_target)
    val_data = lgb.Dataset(val_features, val_target)

    return train_data, val_data


# define function to train model

def train_model(model_name, tuned_params, train_data, val_data):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'feature_pre_filter': False,
        'seed': 0,
        'early_stopping_round': 500,
        'num_iterations': 10000,
        'boosting': 'gbdt',
    }

    params.update(tuned_params)

    classifier = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        verbose_eval=1000,
    )

    joblib.dump(classifier, model_dir_path + model_name + '.joblib')

    return classifier


# define function to load and preprocess test data
def load_test_data():
    transaction_dataframe = pd.read_csv(test_transaction_data_path)
    transaction_dataframe = reduce_mem_usage(transaction_dataframe)

    identity_dataframe = pd.read_csv(test_identity_data_path)
    identity_dataframe = reduce_mem_usage(identity_dataframe)
    identity_dataframe = identity_dataframe.rename(
        columns={
            column: column.replace('-', '_')
            for column in identity_dataframe.columns
        }
    )

    dataframe = transaction_dataframe.merge(identity_dataframe, how='outer')
    transaction_id_data = dataframe['TransactionID']  # need it for output

    del transaction_dataframe
    del identity_dataframe

    print(f'number of rows in test data: {len(dataframe)}')

    test_dataframe = preprocess(dataframe)

    return test_dataframe, transaction_id_data


# define function for doing inference

def inference(output_name, classifier, test_dataframe, transaction_id_data):
    prediction = classifier.predict(test_dataframe)

    output_dataframe = pd.DataFrame({
        'TransactionID': transaction_id_data,
        'isFraud': pd.Series(prediction),
    })

    output_dataframe.to_csv(output_dir_path + output_name + '.csv', index=False)


if __name__ == '__main__':
    os.chdir('..')
    
    print('loading and preprocessing training data...')
    train_data, val_data = load_training_data()

    print('training LightGBM classifier...')
    classifier = train_model(model_name, tuned_params, train_data, val_data)

    del train_data
    del val_data

    print('loading and preprocessing test data...')
    test_dataframe, transaction_id_data = load_test_data()

    print('doing inference on test data...')
    inference(output_name, classifier, test_dataframe, transaction_id_data)
