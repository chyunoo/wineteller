import numpy as np
import pandas as pd
import os

from wineteller.import_data import get_data, clean_wine_data
from wineteller.preprocessor import filtered_mapping

def preprocess_and_train() :

    print("\n⭐️ use case: preprocess and train basic")

    df = get_data("winemag-data_first150k")
    mp= get_data("descriptor_mapping")

    cleaned = clean_wine_data(df)

    fmp = filtered_mapping(mp)



if __name__ == '__main__':
    preprocess_and_train()
