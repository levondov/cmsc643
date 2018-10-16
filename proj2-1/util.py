import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

del_attributes = ['noshow','patient_id','appointment_id'] # attributes not being used

def loadData(fname):
    # read data using read_csv function
    return pd.read_csv(fname)
    
def saveData(fname,dt):
    # save data using to_csv function
    dt.to_csv(fname)
    
def cleanData(dt):
    # rename columns
    dt.columns = ['patient_id','appointment_id','gender','schedule_day','appointment_day', 'age','neighbourhood','scholarship','hipertension','diabetes','alcoholism','handicap','sms_received','noshow']
    # clean up age
    dt = dt.replace(-1,int(dt['age'].mean()))
    # remove useless categories
    dt = dt.drop(del_attributes, axis=1)    
    return dt
    
def createTestData(dt,nsize=20000):
    # shufflesplit data set to create a test set
    split = StratifiedShuffleSplit(n_splits=1, test_size=nsize, random_state=1234)
    for train_index, test_index in split.split(dt, dt["noshow"]):
        train_set = dt.loc[train_index]
        test_set = dt.loc[test_index]
        
    return train_set,test_set
