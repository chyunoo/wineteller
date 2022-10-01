import numpy as np
import pandas as pd
import os

from wineteller.import_data import get_data, clean_wine_data, get_test_data
from wineteller.model import merge_review_vectors, vectorize_reviews, word_embeddings
from wineteller.preprocessor import preprocess_text
from wineteller.registry import save_model, load_model

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

    # train model
    trained_w2v = word_embeddings(df_pp)

    # save model
    save_model(trained_w2v)

    # retrieve wine review vectors
    wine_review_vectors = vectorize_reviews(cleaned, trained_w2v)

    # merge in wine dataset
    vectorized_df = merge_review_vectors(wine_review_vectors, cleaned)

    print(vectorized_df.shape)
    print(vectorized_df.head())


if __name__ == '__main__':
    preprocess_and_train()
