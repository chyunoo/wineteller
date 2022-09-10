import os
import pandas as pd


def get_data():
    csv_path= os.path.join(os.path.dirname(os.path.dirname(__file__)),"raw_data")
    file_names = os.listdir(csv_path)
    data = pd.read_csv(csv_path,file_names)
    return data
