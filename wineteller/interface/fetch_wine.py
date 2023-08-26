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
    #print(f" first output : {recommendations}")
    # to remove from prod
    #recommendations.drop(columns=["review_vector",
    #                              "descriptor_count",
    #                              "description",
    #                              "normalized_descriptors"],
    #                            inplace=True)
    #print(f" second output : {recommendations}")
#
    #recommendations = recommendations.fillna('') # to remove from prod
    #print(f" third output : {recommendations}")
    # recommendations_dict = recommendations.to_dict('records')
    # print(recommendations_dict)
    return recommendations


def preprocess_user_input_old(input : str) :

#     survey = get_data("Survey")
#     cleaned = clean_survey(survey)
#     vectors = vectorize_survey(cleaned)

#     occasion_list = useful_lists()[0]
#     input_tokenized= word_tokenize(input)
#     to_vectorize=[word for word in input_tokenized if word in occasion_list]
#     input_vectorized = np.array([vectors[occasion] for occasion in to_vectorize]).T
#     input_vectorized=np.mean(input_vectorized,axis=1)
#     input_listed =input_vectorized.tolist()

#     return input_listed
    pass

# test compress data
#vectors = np.load('vectorized_survey.npy',allow_pickle='TRUE').item()
#pp_data = get_preprocessed_data("compressed_data") # to compress with pd_numeric
#print(pp_data)
#input_preprocessed = preprocess_user_input("drunk", vectors) # to monitor efficacity
#print(input_preprocessed)
#if input_preprocessed is not None :
#   wine_indices = pair_occasion(input_preprocessed) # to monitor efficacity
#   print(wine_indices)
#   #df = app.state.pp_data
#   df = pp_data # test for baseline
#   pred = fetch_wine(wine_indices, df) # to monitor efficacity
#   print(pred)
