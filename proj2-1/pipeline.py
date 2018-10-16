import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, FunctionTransformer
from sklearn_pandas import DataFrameMapper

cat_attributes = ['gender','handicap','neighbourhood'] # categorical attributes
num_attributes = ['age'] # numerical attributes
bin_attributes = ['scholarship','hipertension','diabetes','alcoholism','sms_received'] # existing binary data attributes


# util custom pipeline transformers
class BinAttribute(BaseEstimator, TransformerMixin):
    # custom class to return already existing binary data
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X
    
class WeekdayTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # for appointment_day attribute
        # convert date to weekday. Sunday - 0 and Saturday - 6 , can later apply labelbinarizer
        newdata = []
        for dt in X:
            newdata.append(datetime.strptime(dt,'%Y-%m-%dT%H:%M:%SZ').weekday())
        return np.array(newdata)

class DaysAheadTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self    
    def transform(self, X, y=None):
        X1 = X[:,0]
        X2 = X[:,1]
        Xfin = []
        for aday,sday in zip(X1,X2):
            d1 = datetime.strptime(aday,'%Y-%m-%dT%H:%M:%SZ')
            d2 = datetime.strptime(sday,'%Y-%m-%dT%H:%M:%SZ')
            Xfin.append((d1-d2).days)
        return np.array(Xfin)
        
#### pipelines

########## dates
weekday_mapper = DataFrameMapper([
    ('appointment_day', WeekdayTransform())
])

daysahead_mapper = DataFrameMapper([
    (['appointment_day', 'schedule_day'], DaysAheadTransform())
])

weekday_pipeline = Pipeline([
    ('weekday_adder', weekday_mapper),
    ('weekday_encoder', OneHotEncoder(categories='auto',sparse=False))
])

daysahead_pipeline = Pipeline([
    ('mapper', daysahead_mapper),
    ('scaler', StandardScaler())
])

date_pipeline = FeatureUnion(transformer_list=[
    ('weekday_pipeline', weekday_pipeline),
    ('daysahead_pipeline', daysahead_pipeline)
])

########## numerical
num_pipeline = Pipeline([
    ('num_attrib', StandardScaler())
])

num_mapper = DataFrameMapper([
    (num_attributes, num_pipeline)
])

########## binary
bin_pipeline = Pipeline([
    ('bin_attrib', BinAttribute())
])

bin_mapper = DataFrameMapper([
    (bin_attributes, bin_pipeline)
])

######### Categorical
cat_mapper = DataFrameMapper([
    ('gender', LabelBinarizer()), # Couldn't figure out a way to iteratively call LabelBinarizer, so have unique call for each attribute
    ('handicap', LabelBinarizer()),
    ('neighbourhood', LabelBinarizer())
])

# full pepeline
full_pipeline = FeatureUnion(transformer_list=[
    ('date_pipeline', date_pipeline),
    ('bin_mapper', bin_mapper),
    ('num_mapper', num_mapper),
    ('cat_mapper', cat_mapper)
])
    
