import numpy as np
import pandas as pd
import os
#pd.set_option('mode.chained_assignment', 'raise')

from wineteller.modeling.import_data import clean_wine_data, get_test_data, get_data
from wineteller.modeling.model import merge_review_vectors, vectorize_reviews, word_embeddings, find_neighbors
from wineteller.modeling.preprocessor import preprocess_text
from wineteller.modeling.registry import save_model, load_model
from wineteller.modeling.vectorize import clean_survey
from wineteller.modeling.preprocessing import preprocess_user_input


def preprocess_and_train() :
    """
    Load wine data, clean data and preprocess it.
    """

    print("\n⭐️ use case: preprocess and train basic")

    # retrieve raw data
    df = get_test_data("winemag-data_first150k")

    # clean data
    cleaned = clean_wine_data(df, keep_columns=True)

    # preprocess data
    df_pp = preprocess_text(cleaned)

    #### remove sentences with zero descriptors ####

    # train model
    trained_w2v = word_embeddings(df_pp)

    # save model
    save_model(trained_w2v)

    # retrieve wine review vectors
    wine_review_vectors = vectorize_reviews(cleaned, trained_w2v)

    # merge with wine dataset
    vectorized_df = merge_review_vectors(wine_review_vectors, cleaned)

    # save into csv
    csv_path= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "raw_data","preprocessed_data", "processed_cols_full.csv")

    vectorized_df.to_csv(csv_path,
                         mode = "w",
                         index=False)

    print(vectorized_df.shape)
    print(vectorized_df.head())

def summary() :
    model = load_model()
    print(model)



if __name__ == '__main__':
    preprocess_and_train()
    summary()
