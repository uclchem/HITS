## %% [markdown]
# # Tracer Identification
# 
# So we have a list of HITs and now we need to find out which ones are useful tracers and what they trace. We'll use RADEX to calculate the line fluxes from the abundances and then train random forests to predict the input variables.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool

from spectralradex import radex

# %% [markdown]
# We first go through all of the useful species and find the collisional data for them. We can check against radex.list_data_files to see what's included with spectral radex and otherwise hunt through LAMDA database.

# %%
data_files=[
    ["CO","co.dat"],
    ["OH","oh.dat"],
    ["HCL","hcl.dat"],
    ["CS","./data/collisional/cs.dat"],
    ["CH","./data/collisional/ch-nohfs.dat"],
    ["HCN","hcn.dat"],
    ["CN","./data/collisional/cn.dat"],
    ["C+","c+.dat"],
    ["HNC","hnc.dat"],
    ["H2O","./data/collisional/ph2o.dat"],
    ["H2O", "./data/collisional/oh2o.dat"],
    ["H3O+","o-h3o+.dat"],
    ["H3O+","p-h3o+.dat"],
    ["NO","./data/collisional/no.dat"],
    ["HCO+","hco+.dat"],
    ["N2H+","./data/collisional/n2h+@xpol.dat"],
    ["NH3","o-nh3.dat"],
    ["NH3","p-nh3.dat"],
    ["CH3CN","./data/collisional/ch3cn.dat"],
    ["H2S","ph2s.dat"],
    ["H2S","oh2s.dat"],
    ["SIO","sio.dat"],
    ["CH2","ch2_h2_ortho.dat"],
    ["CH2","ch2_h2_para.dat"],
    ["HCS+","./data/collisional/hcs+@xpol.dat"],
    ["SO","SO-pH2.dat"],
    #["SO2","so2@lowT.dat"],
    ["H2CO","./data/collisional/oh2co-h2.dat"],
    ["H2CO","./data/collisional/ph2co-h2.dat"],
#    ["OCS","./data/collisional/ocs@xpol.dat"]
]
data_files=pd.DataFrame(columns=["Species","Collisional File"],data=np.asarray(data_files))

# %%
phase="three_phase"
radex.get_default_parameters()

# %%
useful_species=pd.read_csv(f"data/{phase}/useful_species.csv")
useful_species=useful_species.merge(data_files)


# %%
model_data=pd.read_hdf(f"data/{phase}/final_abunds.hdf",)
model_data=model_data[['initialTemp', 'radfield', 'Av', 'zeta','initialDens', 'rout', 'Species',"Abundance"]]
model_data=model_data.groupby(['initialTemp', 'radfield', 'Av', 'zeta','initialDens', 'rout', 'Species'])["Abundance"].mean().reset_index()
model_data=model_data[model_data["Species"].isin(useful_species["Species"])].reset_index(drop=True)

# %%
model_data["NH2"]=model_data["rout"]*3.086e18*model_data["initialDens"]
model_data["cdmol"]=(model_data["Abundance"]*model_data["NH2"])

# %%
model_data["model"]=model_data[["Species","initialTemp","initialDens","cdmol"]].apply(lambda x: f"{x[0]} {x[1]:.0f} {x[2]:.1e} {x[3]:.1e}",axis=1,raw=True)
# %%
run_models=model_data.loc[model_data["model"].drop_duplicates().index]
run_models=run_models.merge(useful_species[["Species","Collisional File"]],on="Species")



# %%
run_models=run_models[run_models["cdmol"]>1e12]


# %%
base_params=radex.get_default_parameters()
base_params["fmin"]=50.0
base_params["fmax"]=1000.0
base_params["linewidth"]=5.0

def get_dict(row):
    params=base_params.copy()
    params["h2"]=row[0]/2.0 #h2 from total H
    params["o-h2"]=0.75*params["h2"]
    params["p-h2"]=0.25*params["h2"]
    params["tkin"]=row[1]
    params["cdmol"]=row[2]
    params["molfile"]=row[3]
    params["species"]=row[4]
    params["model"]=row[5]
    return params

dicts=run_models[["initialDens","initialTemp","cdmol","Collisional File","Species","model"]].apply(get_dict,axis=1,raw=True)


# %%
def radex_helper(param_dict):
    species=param_dict.pop("species")
    model=param_dict.pop("model")
    res=radex.run(param_dict)
    if res is None:
        return pd.DataFrame()
    res["line"]=res["freq"].map(lambda x: f"{species}-{x:.2f}")
    res=res.groupby("line")["FLUX (K*km/s)"].sum().reset_index()
    res["model"]=model
    return res

# %%
with Pool(12) as pool:
    radexes=pool.map(radex_helper,dicts)


# %%
radexes=pd.concat(radexes)
radexes.to_hdf(f"data/{phase}/radex-grid.hdf",key="df",index=False,mode="w")
model_data.to_hdf(f"data/{phase}/model-grid.hdf",key="df",index=False,mode="w")


# %% [markdown]
# # Combine Back
#Now we have the radex fluxes for any given set of N, n and T, we can merge that back on to our model data. Then we'll cut out any lines that overlap (have same line name) because they'll be a pain to deal with. After that it's a simple matter of pivoting out the fluxes so we have a neat table.

# %%
#load flux
radexes=pd.read_hdf(f"data/{phase}/radex-grid.hdf",key="df")
model_data=pd.read_hdf(f"data/{phase}/model-grid.hdf",key="df")
# radexes["line count"]=radexes.groupby(["model","line"]).transform("count")
# radexes=radexes.loc[radexes["line count"]==1,["model","line","FLUX (K*km/s)"]]


# %%
#recover conditions
radexes=radexes.merge(model_data,on="model",how="right")

# %%
radexes
# %%a
idx=radexes["line"]=="NO-250.816"
idx=idx & (radexes["Av"]==10.0)
idx=idx & (radexes["zeta"]==1)
idx=idx & (radexes["radfield"]==1000.0)

idx=idx & (radexes["initialDens"]==100000.0)
radexes.loc[idx]

# %%
radexes=radexes[['line', 'FLUX (K*km/s)', 'initialTemp', 'radfield', 'Av',
       'zeta', 'initialDens', 'rout']]
radexes=radexes.dropna().drop_duplicates()




# %%
radexes=radexes.pivot(index=['initialTemp', 'radfield', 'Av', 'zeta','initialDens', 'rout'],columns="line",values='FLUX (K*km/s)')#.reset_index().fillna(0.0)

# %%
radexes=radexes.reset_index().fillna(0.0)

# %%
radexes.to_hdf(f"data/{phase}/radex-with-conditions.hdf",key="df",index=False)# %%