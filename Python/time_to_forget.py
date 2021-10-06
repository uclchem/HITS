import pandas as pd
import numpy as np
from uclchem import read_output_file
from multiprocessing import Pool
from glob import glob

phase="two_phase"
species=pd.read_csv(f"data/{phase}/useful_species.csv")["Species"].to_list()

completes=glob(f"data/{phase}/models/*.csv")

def get_file(model_file):
    a=read_output_file(model_file)
    a=a[["Time"]+species]
    a=a.melt(id_vars="Time",var_name="Species",value_name="Abundance")
    a["outputFile"]=model_file
    return a

physics=["initialTemp","radfield","Av","zeta","initialDens"]


models=pd.read_csv(f"data/{phase}/models.csv").sort_values(["initialDens","initialTemp","radfield","Av","zeta"])

physics_df=models[physics].drop_duplicates()
physics_df=physics_df.reset_index().rename({"index":"Model"},axis=1)
models=models.merge(physics_df,on=physics)[["Model","outputFile","history"]]

models["outputFile"]=models["outputFile"].str.replace("data/","data/two_phase/")

models=models[models["outputFile"].isin(completes)]


with Pool(32) as pool:
    abundance_df=pool.map(get_file,models["outputFile"])

abundance_df=pd.concat(abundance_df)
abundance_df=abundance_df.reset_index(drop=True)
abundance_df["Abundance"]=np.log10(abundance_df["Abundance"])
models=models.merge(abundance_df,on="outputFile")
del abundance_df

#Only keep the time steps where all 6 histories have the same time.
models=models[models.groupby(["Model","Time","Species"])["history"].transform("count")==6]

#Now work out the between history variances.
models=models.groupby(["Model","Time","Species"])["Abundance"].var().reset_index()
#Sort backwards so we can using an expanding groupby to look at maximum variance for all future times
#also get minimum time of all future times ie the time of the current row.
models=models.sort_values("Time",ascending=False)
models=models.groupby(["Model","Species"]).expanding().agg({"max","min"}).reset_index() #simply the nasty multi-level result
models=models[[(    'Model',    ''),(  'Species',    ''),(     'Time', 'min'),('Abundance', 'max')]]

#end up with a dataframe of the modell, the species and the maximum future variance at every time.
models.columns=["Model","Species","Time","Max Var"]

#Find the minimum time where a species abundance variance is always less than 0.5 dex
models=models[models["Max Var"]<0.5]
models=models.groupby(["Model","Species"])["Time"].min().reset_index()

models.to_csv(f"data/{phase}/time_to_forget.csv",index=False)