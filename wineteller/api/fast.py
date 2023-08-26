from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


from wineteller.interface.fetch_wine import fetch_wine, pair_occasion, preprocess_user_input
from wineteller.modeling.vectorize import vectorize_survey, clean_survey
from wineteller.modeling.import_data import get_data, get_preprocessed_data

#survey = get_data("Survey")
#cleaned = clean_survey(survey)

app = FastAPI()
#app.state.vectorized_survey = vectorize_survey(cleaned) #to remove from prod
#app.state.pp_data = get_preprocessed_data("processed_cols_full_final") # to remove from prod


# test for baseline
#vectorized_survey = vectorize_survey(cleaned) # to pre calculate
pp_data = get_preprocessed_data("compressed_data") # to compress with pd_numeric


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Ok': "Hello wine lover"}

@app.get('/predict')
def predict(occasion : str):

    #vectors = app.state.vectorized_survey
    #vectors = vectorized_survey # test for baseline

    # load pre-calculated survey vectors
    vectors = np.load('vectorized_survey.npy',allow_pickle='TRUE').item()
    input_preprocessed = preprocess_user_input(occasion, vectors) # to monitor efficacity
    if input_preprocessed is not None :
        wine_indices = pair_occasion(input_preprocessed) # to monitor efficacity
        #df = app.state.pp_data
        df = pp_data # test for baseline
        pred = fetch_wine(wine_indices, df) # to monitor efficacity
        return pred
    else :
        return None
