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
    return ax

def _example_generator(ts, 
                      first_target_index=None, 
                      last_target_index=None,
                      horizon=4, npoints=16,
                      batchsize=20,
             norm_mns=None, norm_sds=None):
    if first_target_index is None:
        first_target_index = npoints + horizon
    if last_target_index is None:
        last_target_index = ts.shape[1] - 1
    
    nrows = ts.shape[0]
    
    i=0
    j=first_target_index
    while 1:
        if i + batchsize > nrows:
            rows = np.hstack((np.arange(i,nrows), np.arange(0,batchsize-(nrows-i))))
            i = 0
            j = j + 1 if j < last_target_index - 1 else last_target_index
        else:
            rows = np.arange(i, i + batchsize)
            i += batchsize
        cols = np.arange(j-horizon-npoints, j-horizon)
        features = np.zeros((len(rows), len(cols),1))
        for jj in range(len(cols)):
            features[:,jj,0] = ts[rows, cols[jj]]
            if norm_mns is not None:
                features[:,jj,0] -= norm_mns[rows]
            if norm_sds is not None:
                features[:,jj,0] /= norm_sds[rows]
        targets = np.zeros((len(rows),))
        for ii in range(len(rows)):
            targets[ii] = ts[rows[ii],j]
            if norm_mns is not None:
                targets[ii] -= norm_mns[rows[ii]]
            if norm_sds is not None:
                targets[ii] /= norm_sds[rows[ii]]
        yield features, targets

    
class TSExampleGenerator:
    def __init__(self, df, horizon=4, n_prediction_years=4, 
                 n_test_quarters=4, n_val_quarters=4,
                 normalize=True):
        self._horizon = horizon
        self._nyears = n_prediction_years
        self._ts = df.drop(columns=['RegionID','RegionName','SizeRank']).values
        self._last_test_index = self._ts.shape[1] - 1
        self._last_val_index = self._last_test_index - n_test_quarters
        self._last_train_index = self._last_val_index - n_val_quarters
        self._nregions = self._ts.shape[0]
        self._norm_mns = None
        self._norm_sds = None
        
        if normalize:
            self.set_normalization_parms()
        
    def set_normalization_parms(self, norm_mns=None, norm_sds=None):
        train_ts = self._ts[:,:self._last_train_index]
        self._norm_mns = train_ts.mean(axis=1) if norm_mns is None else norm_mns
        self._norm_sds = train_ts.std(axis=1) if norm_sds is None else norm_sds

    def _check_norm(self):
        assert (self._norm_mns is not None), "Normalization params not set, call set_normalization_parms method"
        assert (self._norm_sds is not None), "Normalization params not set, call set_normalization_parms method"

    def get_train_gen(self, batchsize=20, normalize=True):
        if normalize:
            self._check_norm()
            
        gen = _example_generator(self._ts, 
                                    last_target_index=self._last_train_index,
                                    norm_mns=self._norm_mns if normalize else None, 
                                    norm_sds=self._norm_sds if normalize else None,
                                    batchsize=batchsize)
        num_sequences = self._nregions * self._last_train_index
        num_steps = int(np.ceil(num_sequences / batchsize))
        return gen, num_steps

    def get_val_gen(self, batchsize=20, normalize=True):
        if normalize:
            self._check_norm()
            
        gen = _example_generator(self._ts, 
                                first_target_index = self._last_train_index+1,
                                last_target_index=self._last_val_index,
                                norm_mns=self._norm_mns if normalize else None, 
                                norm_sds=self._norm_sds if normalize else None,
                                batchsize=batchsize)
        num_sequences = self._nregions * (self._last_val_index - self._last_train_index) 
        num_steps = int(np.ceil(num_sequences / batchsize))
        return gen, num_steps
    
    def get_test_gen(self, batchsize=20, normalize=True):
        if normalize:
            self._check_norm()
            
        gen = _example_generator(self._ts, 
                                first_target_index=self._last_val_index+1,
                                norm_mns=self._norm_mns if normalize else None, 
                                norm_sds=self._norm_sds if normalize else None,
                                batchsize=batchsize)
        num_sequences = self._nregions * (self._last_test_index - self._last_val_index)
        num_steps = int(np.ceil(num_sequences / batchsize))        
        return gen, num_steps
