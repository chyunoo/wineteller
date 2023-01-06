import os
import pandas as pd
import pathlib
import numpy as np

def get_data(name : str) -> pd.DataFrame :
    """
    retrieve raw data (wine reviews or wine mapping or survey)
    """

    name = name+".csv"
    csv_path= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"raw_data",name)
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
    return data


def clean_wine_data(data : pd.DataFrame, keep_columns = False) -> pd.DataFrame:
    """
    clean raw wine data by removing unnecessary columns and duplicates,
    and keep only description column
    """
    if keep_columns == False :
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

    else :
        #Let's keep all the relevant columns
        data.drop(columns = ["region_2",
                             "points",
                             "price",
                             "designation",
                             "winery",
                             "Unnamed: 0"], inplace=True)
        data = data.drop_duplicates()
        print(data.head())
        print(f"\n✅ data cleaned : {data.shape}")
        print(data.columns)

        return data


def get_preprocessed_data(name) :
    name = name+".csv"
    csv_path= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"raw_data", "preprocessed_data", name)

    preprocessed = pd.read_csv(csv_path)
    df_mincount = preprocessed[preprocessed["descriptor_count"]>0]
    df_mincount["review_vector"]=[np.float_(i[2:-2].split()) for i in df_mincount["review_vector"]]

    return df_mincount
