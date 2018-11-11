import pandas as pd
import numpy as np

def load_df():
    return pd.read_csv("data/Affordability_Wide_2017Q1_Public.csv")

def process_df(df):
    new_df = df[(df['Index'] == "Mortgage Affordability") & (df['SizeRank'] != 0)]\
        .drop(columns=['Index','HistoricAverage_1985thru1999'])\
        .dropna()
    return new_df

def get_affordability_df():
    return process_df(load_df())

def plot_affordability_series(df):
    plotdf = df.drop(columns=['RegionName','SizeRank']).\
        melt(id_vars=['RegionID'], var_name='date', value_name='affordability').\
        pivot(index='date', columns='RegionID', values='affordability')
    ax = plotdf.plot(legend=False, color='lightgray')
    ax.set_xlabel('Date')
    ax.set_ylabel('Affordability')
    ax.set_title("Mortgage Affordability")
    
    return plotdf

    

def split_train_test(df, nyears=2):
    lag = nyears * 4 + 1
    
    metadata_columns = ['RegionID', 'RegionName', 'SizeRank']
    metadata = df[metadata_columns]
    ts = df.drop(columns=metadata_columns)
    
    train = pd.concat([metadata, ts.iloc[:,:-lag]], axis=1)
    test = pd.concat([metadata, ts.iloc[:,-lag:]], axis=1)
    return train, test

def get_X_y(df, lag=4):        
    dat = df.drop(columns=['RegionName', 'SizeRank', 'RegionID']).values
    n, m = dat.shape
    
    tmp = np.empty([0, lag+1])
    
    for i in range(n):
        for j in range(m - lag - 1):
            x = dat[i, j:(j+lag+1)]
            tmp = np.vstack([tmp, x])
    X = tmp[:,:-1]
    y = tmp[:, -1]
    return X, y

def get_rnn_dat(df):
    dat = df.drop(columns=['RegionName', 'SizeRank', 'RegionID']).values
    x = dat[:, :-1]
    y = dat[:, 1:]
    return x, y

def get_rnn_chunker(x, y, batch_size=10, chunk_size=8):
    n, m = x.shape
    r = m % chunk_size
    num_chunks = (m // chunk_size)
    pad_length = chunk_size - r
    
    def chunker(chunk, batch):
        batch_start = batch * batch_size
        batch_end = batch_start + batch_size
        
        chunk_start = chunk * chunk_size
        chunk_end = min([chunk_start + chunk_size, m])
        
        xx = x[batch_start:batch_end,chunk_start:chunk_end]
        yy = y[batch_start:batch_end,chunk_start:chunk_end]
        
        seq_length = np.full((batch_size), chunk_size)  
        
        if chunk == num_chunks - 1 and r > 0:        
            pad = np.zeros((cur_batch_size, pad_length))
            xx = np.concatenate([xx, pad], axis=1)
            yy = np.concatenate([yy, pad], axis=1)
            seq_length = np.full((batch_size), r)
            
        xx = xx.reshape([batch_size, chunk_size, 1])
        yy = yy.reshape([batch_size, chunk_size, 1])
        return xx, yy, seq_length
    return chunker
    
