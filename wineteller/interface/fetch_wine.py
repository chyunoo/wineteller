from wineteller.modeling.import_data import get_preprocessed_data
from wineteller.modeling.model import find_neighbors
from wineteller.modeling.vectorize import useful_lists
from wineteller.modeling.params import MAPPING
from nltk.tokenize import word_tokenize

import nltk
#nltk.download('punkt')

import numpy as np
import pandas as pd


def preprocess_user_input(input : str, vectors) :

    occasion_list = useful_lists()[0]
    input_tokenized= word_tokenize(input)
    to_vectorize=[word for word in input_tokenized if word in occasion_list]
    if len(to_vectorize) > 0 :
        input_vectorized = np.array([vectors[occasion] for occasion in to_vectorize]).T
        input_vectorized=np.mean(input_vectorized,axis=1)
        input_listed =input_vectorized.tolist()
        return input_listed
    else :
        return None

def pair_occasion(input_listed : list) :

    wine_indices = find_neighbors(input_listed)
    return wine_indices


def fetch_wine(wine_indices : list, df : pd.DataFrame) :

    recommendations = df.loc[df.index[wine_indices]]
    print(f'{recommendations=}')
    return recommendations


def preprocess_user_input_old(input : str) :
    pass
