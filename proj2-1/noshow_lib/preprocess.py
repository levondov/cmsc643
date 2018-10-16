# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler

from sklearn_pandas import DataFrameMapper
import noshow_lib.util as utils

from sklearn.externals import joblib

class WeekdayTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X['AppointmentDay'].dt.weekday.values
    

class DaysAheadTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        appt = X['AppointmentDay']
        sched = X['ScheduledDay']
        days_ahead = (appt - sched).dt.days.values.astype('float64')
        return days_ahead
    
def get_features_pipeline():
    weekday_mapper = DataFrameMapper([
            (['AppointmentDay'], WeekdayTransform())
        ], 
        input_df=True)

    weekday_pipeline = Pipeline([
            ('weekday_adder', weekday_mapper),
            ('weekday_encoder', OneHotEncoder(n_values=7))
        ]
    )
    
    days_ahead_mapper = DataFrameMapper([
            (['AppointmentDay', 'ScheduledDay'], DaysAheadTransform())
        ], 
        input_df=True
    )
    
    days_ahead_pipeline = Pipeline([
            ('days_ahead_mapper', days_ahead_mapper),
            ('days_ahead_scaler', StandardScaler())
        ]
    )
    
    pass_through_attributes =\
        ['Scholarship',
         'Hypertension',
         'Diabetes',
         'Alcoholism',
         'SMS_received']    
    
    pass_through_mapper = DataFrameMapper(
        list(zip(
            pass_through_attributes, 
            [None for x in pass_through_attributes]
        ))
    )
        
    mapper = DataFrameMapper([
        (['Age'], StandardScaler()),
        ('Gender', LabelBinarizer()),
        (['Handicap'], OneHotEncoder(n_values=5)),
        ('Neighbourhood', LabelBinarizer())
    ])
    
    full_pipeline = FeatureUnion(transformer_list=[
        ('weekday_pipeline', weekday_pipeline), # add weekday feature
        ('days_ahead_pipeline', days_ahead_pipeline), # add days ahead feature
        ('pass_through_mapper', pass_through_mapper), # pass 5 features through
        ('mapper', mapper) # age (1), gender (1), handicap (4), neighourhood (?)
    ])
    
    return full_pipeline
            
def get_labels_pipeline():
    mapper = DataFrameMapper([
        ('No-show', LabelBinarizer(pos_label=1, neg_label=-1))
    ])
        
    return mapper

def fit_save_pipelines(config=utils.file_config):
    train_df = utils.read_csv(config['processed_data_path'], 
                              config['train_csv'])
    
    # fit feature pipeline
    feature_pipeline = get_features_pipeline()
    feature_pipeline = feature_pipeline.fit(train_df)
    
    # save feature pipeline after fitting
    joblib.dump(feature_pipeline, 
                config['objstore_path'] + '/' + config['feature_pipeline_file'])
    
    # fit labels pipeline
    label_pipeline = get_labels_pipeline()
    label_pipeline = label_pipeline.fit(train_df)
    
    # save labels pipeline
    joblib.dump(label_pipeline, 
                config['objstore_path'] + '/' + config['labels_pipeline_file'])
    
def load_pipelines(config=utils.file_config):
    feature_pipeline_path = config['objstore_path'] + '/' +\
        config['feature_pipeline_file']
    feature_pipeline = joblib.load(feature_pipeline_path)
    
    labels_pipeline_path = config['objstore_path'] + '/' +\
        config['labels_pipeline_file']
        
    label_pipeline = joblib.load(labels_pipeline_path)
    return feature_pipeline, label_pipeline

def preprocess_data(data_frame, config=utils.file_config):
    feature_pipeline, labels_pipeline = load_pipelines(config=config)
    X = feature_pipeline.transform(data_frame).toarray()

    nobs = X.shape[0]
    y = labels_pipeline.transform(data_frame).reshape((nobs))
    return X, y

def load_train_data(config=utils.file_config):
    train_df = utils.read_csv(config['processed_data_path'], 
                              config['train_csv'])
    train_X, train_y = preprocess_data(train_df, config=config)
    return train_X, train_y

def load_test_data(config=utils.file_config):
    test_df = utils.read_csv(config['processed_data_path'], 
                             config['test_csv'])
    test_X, test_y = preprocess_data(test_df, config=config)
    return test_X, test_y
    
    
