import pandas as pd
import numpy as np
from uclchem import read_output_file
from multiprocessing import Pool

phase="three_phase"
species=pd.read_csv(f"data/{phase}/useful_species.csv")["Species"].to_list()

def get_file(model_file):
    a=read_output_file(model_file)
    a=a[["Time",'CO', 'OH', 'H2O', 'NH2', 'CN', 'NH3', 'HCO', 'NH', 'HCL', 'H3O+',
       'CH', 'CH3', 'HCN', 'NO', 'HCO+', 'CH2', 'N2H+', 'H2CO', 'HNC',
       'CH3CN']]
    a=a.melt(id_vars="Time",var_name="Species",value_name="Abundance")
    a["outputFile"]=model_file
    return a

physics=["initialTemp","radfield","Av","zeta","initialDens"]
models=pd.read_csv(f"data/{phase}/models.csv")[physics].drop_duplicates()

# low density
idx=(models["initialDens"]==1e4) & (models["zeta"]<3) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=models.loc[idx].sample(1)

idx=(models["initialDens"]==1e4) & (models["zeta"]<3) & (models["initialTemp"]>200)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=plot_models.append(models.loc[idx].sample(1))


idx=(models["initialDens"]==1e4) & (models["zeta"]>100) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=plot_models.append(models.loc[idx].sample(1))

idx=(models["initialDens"]==1e4) & (models["zeta"]<5) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]>500) & (models["Av"]<5)
plot_models=plot_models.append(models.loc[idx].sample(1))


#high density
idx=(models["initialDens"]==1e7) & (models["zeta"]<3) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=plot_models.append(models.loc[idx].sample(1))

idx=(models["initialDens"]==1e7) & (models["zeta"]<3) & (models["initialTemp"]>200)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=plot_models.append(models.loc[idx].sample(1))

idx=(models["initialDens"]==1e7) & (models["zeta"]>100) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]<7) & (models["radfield"]*np.exp(-3.02*models["Av"])<1)
plot_models=plot_models.append(models.loc[idx].sample(1))

idx=(models["initialDens"]==1e7) & (models["zeta"]<5) & (models["initialTemp"]<30)
idx = idx & (models["radfield"]>500) & (models["Av"]<5)
plot_models=plot_models.append(models.loc[idx].sample(1))

models=pd.read_csv(f"data/{phase}/models.csv")

plot_models=models.merge(plot_models,on=["initialTemp","radfield","Av","zeta","initialDens"],how="inner")
print(len(plot_models))
plot_models=plot_models.reset_index(drop=True)


with Pool(32) as pool:
    abundance_df=pool.map(get_file,plot_models["outputFile"])

abundance_df=pd.concat(abundance_df)
abundance_df=abundance_df.reset_index(drop=True)
plot_models=plot_models.merge(abundance_df,on="outputFile")
plot_models=plot_models.drop("outputFile",axis=1)
plot_models.to_hdf(f"data/{phase}/plot_models.hdf",index=False,key="df",mode="w")
