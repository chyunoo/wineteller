import numpy as np
import pandas as pd
import os

from wineteller.import_data import get_data, clean_wine_data, get_test_data
from wineteller.preprocessor import preprocess_text

def preprocess_and_train() :
    """
    Load wine data, clean data and preprocess it.
    """

    print("\n⭐️ use case: preprocess and train basic")

    # retrieve raw data
    df = get_test_data("winemag-data_first150k")

    # clean data
    cleaned = clean_wine_data(df)

    # preprocess data
    df_pp = preprocess_text(cleaned)

    return df_pp

if __name__ == '__main__':
    preprocess_and_train()
