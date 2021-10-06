import uclchem
import numpy as np
import pandas as pd
from schwimmbad import MPIPool
import time



with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)


    #basic set of parameters we'll use for this grid. 
    ParameterDictionary = {"phase": 1, "switch": 0, "collapse": 1, "readAbunds": 0,
                            "fr":1.0,"rout":0.1,"zeta":1.0,"radfield":1.0,
                           "outSpecies": 'SO CO',"finalTime":1.0e6,"baseAv":1}


        # This part can be substituted with any choice of grid
        # here we just combine various initial and final densities into an easily iterable array
    grid={
        "initialTemp":[10.0,30.0],
        "initialDens":[100.0,"final"],
        "finalDens":np.logspace(4,7,4),
        }

    #generate grid of all combinations
    parameterSpace = np.asarray(np.meshgrid(*list(grid.values()))).reshape(len(grid.keys()), -1)
    history_df=pd.DataFrame(columns=grid.keys(),data=parameterSpace.T)

    #remove combinations we don't want
    idx=(history_df["initialDens"]=="final") & (history_df["initialTemp"]==30.0)
    history_df=history_df.loc[~idx]

    history_df["initialDens"]=np.where(history_df["initialDens"]=="final",history_df["finalDens"],history_df["initialDens"])
    history_df["outputFile"]=history_df.index.map(lambda x:f"data/three_phase/histories/{x+1:.0f}.csv")

    for key,value in grid.items():
        history_df[key]=history_df[key].astype(float)
    history_df.to_csv("data/three_phase/histories.csv",index=False)


    models=[]
    for i,row in history_df.iterrows():
        paramDict=ParameterDictionary.copy()
        for key,value in row.iteritems():
            paramDict[key]=value
        models.append(paramDict)
        print(paramDict)
    #use pool.map to run each dictionary throuh our helper function
    start=time.perf_counter()
    result=pool.map(uclchem.phase_one,models)
    result=np.asarray(result)
    pool.close()
    end=time.perf_counter()
    end=(end-start)/60.0
    print(f"grid in {end} minutes")

    history_df["collapse"]=np.where(history_df["initialDens"]!=history_df["finalDens"],1,0)
    history_df["long"]=np.where(history_df["initialDens"]!=history_df["finalDens"],1,0)
    i=history_df.index.max()+2
    for j,row in history_df.iterrows():
        if ((row["collapse"]+row["long"])==2):
            if row["initialTemp"]==10.0:
                model_df=pd.read_csv(row["outputFile"],skiprows=2)
                row["outputFile"]=f"data/three_phase/histories/{i:.0f}.csv"
                row["long"]=0
                model_df=model_df.loc[0:model_df[model_df["Density"]==model_df["Density"].max()].index[0]]
                model_df.to_csv(row["outputFile"],index=False)
                history_df.loc[len(history_df)]=row
                i=i+1
        elif (row["collapse"]==0.0) & (row["initialTemp"]==10.0):
            if row["outputFile"] is not None:
                row["outputFile"]=None
                history_df.loc[len(history_df)]=row

    history_df.to_csv("data/three_phase/histories.csv",index=False)
