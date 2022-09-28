import os
import pandas as pd
import pathlib

def get_data(name : str) -> pd.DataFrame :
    """
    retrieve raw data (wine reviews or wine mapping)
    """

    name = name+".csv"
    csv_path= os.path.join(os.path.dirname(os.path.dirname(__file__)),"raw_data",name)
    data = pd.read_csv(csv_path)

    return data

def get_test_data(name):
    """
    retrieve raw test data (e.g sample of wine reviews)
    """

    print(" importing test data ... ")
    name = name+".csv"
    csv_path= os.path.join(pathlib.Path().absolute(),"raw_data",name)
    data = pd.read_csv(csv_path)
    return data[:1000]


def clean_wine_data(data : pd.DataFrame) -> pd.DataFrame:
    """
    clean raw wine data by removing unnecessary columns and duplicates,
    and keep only description column
    """

    data.drop(columns = ["region_1",
                         "region_2",
                         "points",
                         "price",
                         "designation",
                         "winery",
                         "Unnamed: 0"], inplace=True)
    data = data.drop_duplicates()
    #Let's keep only the wine descriptions
    data = data[["description"]]
    print(data.head())
    print(f"\n✅ data cleaned : {data.shape}")

    return data
