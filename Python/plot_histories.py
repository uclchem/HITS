import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import uclchem
from glob import glob

history_files=glob("data/histories/*.csv")
species=["H","H2","CO","#CO","H2O","#H2O","CH3OH","#CH3OH"]

for history_file in history_files:
    fig,ax=plt.subplots(figsize=(16,9),tight_layout=True)
    try:
        df=uclchem.read_output_file(history_file)
    except:
        df=pd.read_csv(history_file)
        df.columns=df.columns.str.strip()
    output_file=history_file.replace("histories","history-plots")
    output_file=history_file.replace("csv","png")

    uclchem.plot_species(ax,df,species)
    fig.savefig(output_file,dpi=300)