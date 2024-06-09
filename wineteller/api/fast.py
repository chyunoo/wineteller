from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


from wineteller.interface.fetch_wine import fetch_wine, pair_occasion, preprocess_user_input
from wineteller.modeling.import_data import get_data, get_preprocessed_data

app = FastAPI()
pp_data = get_preprocessed_data("compressed_data_old")

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Ok': "Hello wine lover"}

@app.get('/predict')
def predict(occasion : str):

    # load pre-calculated survey vectors
    vectors = np.load('/Users/hyunoochang/code/chyunoo/wineteller/raw_data/old/vectorized_survey_old.npy',allow_pickle='TRUE').item()
    input_preprocessed = preprocess_user_input(occasion, vectors)
    if input_preprocessed is not None :
        wine_indices = pair_occasion(input_preprocessed)
        print(f'{wine_indices=}')
        df = pp_data
        pred = fetch_wine(wine_indices, df)
        print(f'{pred=}')
        return pred
    else :
        return None
