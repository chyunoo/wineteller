from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wineteller.interface.fetch_wine import fetch_wine, pair_occasion, preprocess_user_input
from wineteller.modeling.vectorize import vectorize_survey, clean_survey
from wineteller.modeling.import_data import get_data, get_preprocessed_data

survey = get_data("Survey")
cleaned = clean_survey(survey)

app = FastAPI()
app.state.vectorized_survey = vectorize_survey(cleaned)
app.state.pp_data = get_preprocessed_data("processed_cols_full_final")

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

    vectors = app.state.vectorized_survey
    input_preprocessed = preprocess_user_input(occasion, vectors)
    if input_preprocessed is not None :
        wine_indices = pair_occasion(input_preprocessed)
        df = app.state.pp_data
        pred = fetch_wine(wine_indices, df)
        return pred
    else :
        return None
