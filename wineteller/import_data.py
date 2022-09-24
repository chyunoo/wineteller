import os
import pandas as pd
import pathlib

def get_data(name):
    name = name+".csv"
    csv_path= os.path.join(pathlib.Path().absolute(),"raw_data",name)
    data = pd.read_csv(csv_path)

    return data


def clean_wine_data(data) :
    #Drop unnecesseary columns
    data.drop(columns = ["region_1",
                         "region_2",
                         "points",
                         "price",
                         "designation",
                         "winery",
                         "Unnamed: 0"], inplace=True)
    #Drop duplicates
    data = data.drop_duplicates()

    #Let's keep only the reviews
    data = data[["description"]]
    print(data.head())
    print(f"\n✅ data cleaned : {data.shape}")

    return data
