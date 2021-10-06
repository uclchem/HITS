#
# This script takes all the model outputs and creates a dataframe with the final abundance of every species for every physics
# More useful than the final abundances.
#
from numpy.core.fromnumeric import compress
import pandas as pd
from glob import glob
from uclchem import read_output_file,check_abunds
from multiprocessing import Pool
from os import remove

def read_last_abunds(data_file):

    a=read_output_file(data_file)
    if (a["Time"].max()>9.99e5):
        a=a.iloc[-1,5:-2].reset_index()
        a.columns=["Species","Abundance"]
        a["outputFile"]=data_file
        a["Conserve C"]=check_conserve(data_file)
    else:
        a=pd.DataFrame()
    return a

def check_conserve(data_file):
    a=read_output_file(data_file)
    result=check_abunds("C",a)
    result=(result.iloc[0]-result.iloc[-1])/result.iloc[0]
    return abs(result)<0.01

def remover(file_name):
    try:
        remove(file_name)
    except:
        pass


phase="three_phase"
models=pd.read_csv(f"data/{phase}/models.csv")
#not all completed!
models=models[models["outputFile"].isin(glob(f"data/{phase}/models/*"))]

with Pool(63) as pool:
    a=pool.map(read_last_abunds,models["outputFile"].values)
    a=pd.concat(list(a))

models=models.merge(a,on="outputFile")
models=models.dropna()
models.to_hdf(f"data/{phase}/final_abunds.hdf",key="df",mode="w")
completed=models["outputFile"].unique()

bad_c=list(models.loc[models["Conserve C"]==False,"outputFile"].unique())

models=pd.read_csv(f"data/{phase}/models.csv")
idx=models["outputFile"].isin(completed)
incomplete=list(models.loc[~idx,"outputFile"].unique())

removes=pd.Series(incomplete+bad_c).unique()
pd.Series(removes).map(remover)