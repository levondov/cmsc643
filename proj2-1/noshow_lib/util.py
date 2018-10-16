# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


file_config = {'raw_data_path': "data",
               'raw_data_csv': "KaggleV2-May-2016.csv",
               'processed_data_path': "processed_data",
               'train_csv': "train_set.csv",
               'test_csv': "test_set.csv",
               'objstore_path': "objects",
               'feature_pipeline_file': "feature_pipeline.pkl",
               'labels_pipeline_file': "labels_pipeline.pkl"}

def read_csv(path, file):
    df = pd.read_csv(path + "/" + file,
                     parse_dates = ['AppointmentDay', 'ScheduledDay'],
                     dtype = {'Age': np.float64})
    return df

def read_kaggle_csv(config=file_config):
    return read_csv(config['raw_data_path'], config['raw_data_csv'])
    
def read_training_csv(config=file_config):
    return read_csv(config['processed_data_path'], 
                    config['train_csv'])

def read_testing_csv(config=file_config):
    return read_csv(config['processed_data_path'], 
                    config['test_csv'])

def make_train_test_sets(config=file_config,
                         test_size = 20000,
                         random_state = 1234):
    kaggle_df = read_kaggle_csv(config)
    
    # rename some variables
    kaggle_df = kaggle_df.rename(index = str,
                     columns = {"Hipertension": "Hypertension",
                                "Handcap": "Handicap"}
                     )
    
    # remove negative ages
    rows_to_drop = kaggle_df['Age'] < 0
    kaggle_df = kaggle_df.drop(kaggle_df[rows_to_drop].index)
    
    split = StratifiedShuffleSplit(n_splits=1, 
                                   test_size=test_size,
                                   random_state = random_state)
    
    for train_index, test_index in split.split(kaggle_df, kaggle_df['No-show']):
        train_set = kaggle_df.iloc[train_index]
        test_set = kaggle_df.iloc[test_index]
        
    train_path = config['processed_data_path'] + '/' + config['train_csv']
    train_set.to_csv(train_path, index=False)
    
    test_path = config['processed_data_path'] + '/' + config['test_csv']
    test_set.to_csv(test_path, index=False)
    