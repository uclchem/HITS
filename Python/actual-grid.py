#!/easybuild/easybuild/el7/software/Miniconda3/4.7.10/bin/python
#Marcus Keil and Jon Holdship 13/03/2020
#Examples of a simple grid of models run in parallel
import uclchem
import numpy as np
import pandas as pd
from schwimmbad import MPIPool
import time
from glob import glob
from pyDOE import lhs

phases="three_phase"
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    generate_grid=False
    run_grid=True
    #basic set of parameters we'll use for this grid. 
    ParameterDictionary = {"phase": 1, "switch": 0, "collapse": 0, "readAbunds": 0,
                            "fr":1.0, "metallicity":1.0,
                           "outSpecies": 'SO CO',"finalTime":1.0e6,"baseAv":1}

    history_df=pd.read_csv(f"data/{phases}/histories.csv")[["outputFile","finalDens"]].rename({"outputFile":"history","finalDens":"initialDens"},axis=1)
    if generate_grid:
            # This part can be substituted with any choice of grid
            # here we just combine various initial and final densities into an easily iterable array
        parameter_limits={
            "initialTemp":[10.0,300.0],
            "radfield":[0,3],
            "zeta":[0,3],
            "Av":[1,10],
        }
        log_sampled=["radfield","zeta"]
        models=[]

        samples=lhs(len(parameter_limits),1000,criterion="center")
        parameter_df=pd.DataFrame(columns=parameter_limits.keys(),data=samples)
        for column,limits in parameter_limits.items():
            parameter_df[column]=(parameter_df[column].values*(limits[1]-limits[0]))+limits[0]
            if column in log_sampled:
                parameter_df[column]=10.0**parameter_df[column]

        histories=history_df.index
        temp_df=parameter_df.copy()
        parameter_df["history_index"]=histories[0]
        for history in histories[1:]:
            temp_df["history_index"]=history
            parameter_df=parameter_df.append(temp_df)
        parameter_df=parameter_df.reset_index(drop=True)
        parameter_df["outputFile"]=parameter_df.index.map(lambda x:f"data/{phases}/models/{x+1:.0f}.csv")
        parameter_df=parameter_df.merge(history_df,left_on="history_index",right_index=True,how="left")
        parameter_df=parameter_df.drop("history_index",axis=1)
        parameter_df["history"]=np.where(parameter_df["history"].isna(),"No History",parameter_df["history"])

        parameter_df["rout"]=(parameter_df["Av"]-ParameterDictionary["baseAv"])*1.6e21
        parameter_df["rout"]=parameter_df["rout"]/(parameter_df["initialDens"]*3.086e18)+0.0001
        parameter_df["rout"]=parameter_df["rout"].round(4)


        for key,value in parameter_limits.items():
            if key != "history_index":
                parameter_df[key]=parameter_df[key].astype(float)

        parameter_df.to_csv(f"data/{phases}/models.csv",index=False)
    else:
        parameter_df=pd.read_csv(f"data/{phases}/models.csv")
    if run_grid:
        models=[]
        finished_models=glob(f"data/{phases}/models/*.csv")
        print(len(finished_models))
        print(finished_models[0])
        for i,row in parameter_df.iterrows():
            if row["outputFile"] not in finished_models:
                paramDict=ParameterDictionary.copy()
                for key,value in row.iteritems():
                    if key!="Av":
                        paramDict[key]=value
                models.append(paramDict)
        #use pool.map to run each dictionary throuh our helper function
        start=time.perf_counter()
        result=pool.map(uclchem.phase_two,models)
        pool.close()
        end=time.perf_counter()
        end=(end-start)/60.0
        print(f"grid in {end} minutes")

