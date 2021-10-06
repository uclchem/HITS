import pandas as pd

class TableManager:
    def __init__(self,table_file):
        self.master_table=pd.read_csv(table_file).sort_values("Info",ascending=False)

    def get_filtered_table(self,low_freq,high_freq,delta_freq,target):
        low_freq=float(low_freq)
        high_freq=float(high_freq)
        delta_freq=float(delta_freq)
        table=self.master_table[self.master_table["Target"]==target]
        print(table.columns)
        table=table[table["Freq 1"]>low_freq]
        table=table[table["Freq 1"]<high_freq]
        table=table[table["Freq 2"]>low_freq]
        table=table[table["Freq 2"]<high_freq]
        table=table[(table["Freq 1"]-table["Freq 2"]).abs()<delta_freq]
        return table[["Feature","Info"]]