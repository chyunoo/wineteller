import os
import pandas as pd
import pathlib

def get_data(name):
    name = name+".csv"
    csv_path= os.path.join(os.path.dirname(pathlib.Path().absolute()),"raw_data",name)
    data = pd.read_csv(csv_path)
    data = data.dropna()
    return data
