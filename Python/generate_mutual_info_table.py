import pandas as pd
import numpy as np

from multiprocessing import Pool
from pandas.io.pytables import read_hdf
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import mutual_info_regression

from itertools import permutations


#%% 
'''
# Data load and set up
We need to load up our RADEX grid, grab all the transitions and physical variables as two separate lists and 
add some random noise to the transitions so that extremely weak transitions are not useful.
'''
if __name__ == '__main__':

    phase="three_phase"
    df=pd.read_hdf(f"../data/{phase}/radex-with-conditions.hdf",key="df").reset_index(drop=True)

    physics=['initialTemp', 'radfield', 'Av', 'zeta', 'initialDens', 'rout']
    df["initialDens"]=np.log10(df["initialDens"])
    df["radfield"]=np.log10(df["radfield"])
    df["zeta"]=np.log10(df["zeta"])

    features=[x for x in df.columns if x not in physics]


    rms=0.05
    delta_v=.5
    n_channels=15.0
    noise=rms*delta_v*np.sqrt(n_channels)
    add_noise = lambda x: x+np.random.normal(0,noise)
    df[features]=df[features].apply(add_noise)
    transformer=QuantileTransformer()
    df[features]=transformer.fit_transform(df[features])
    df=df+0.001

    '''
    Then we generate all permuations to get ratios
    '''
    print("Starting Ratios")
    def get_all_ratios(column):
        res=df.divide(df[column],axis='index')
        res.columns=res.columns.map(lambda x: f"RATIO({x}/{column})")
        return res


    with Pool(64) as pool:
        result=pool.map(get_all_ratios,features)
        pool.close()
        pool.join()
    df=[df]+result
    df=pd.concat(df,axis=1)
    df.to_hdf(f"../data/{phase}/radex_with_ratios.hdf",index=False,key="df")
    del result
    del df
    
    df=pd.read_hdf(f"../data/{phase}/radex_with_ratios.hdf",key="df")
    df=df.replace([np.inf, -np.inf], np.nan)
    features=[x for x in df.columns if x not in physics]

    '''
    All that remains is to calculate the mutual information between every feature and every variable
    '''
    
    print("Starting Mutual Info")
    def info_from_pair(pair):
        print(pair)
        try:
            idx=df[list(pair)].notna().all(axis=1)
            return mutual_info_regression(df.loc[idx,pair[0]].values.reshape(-1, 1),df.loc[idx,pair[1]])[0]
        except:
            return 0




    pairs=[(feature,variable) for feature in features for variable in physics]
    with Pool(64) as pool:
        result=pool.map(info_from_pair,pairs)
        pool.close()
        pool.join()
    features=[pair[0] for pair in pairs]
    variables=[pair[1] for pair in pairs]
    feature_df=pd.DataFrame({"Feature":features,"Target":variables,"Info":result})
    feature_df.to_csv(f"../data/{phase}/feature_info_table.csv",index=False)