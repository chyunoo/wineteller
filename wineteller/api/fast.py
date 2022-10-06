from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wineteller.interface.fetch_vector import similar_word
from wineteller.modeling.registry import load_model


app = FastAPI()

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
def predict(word : str,
            n : int):


    pred = similar_word(word,n)

    return dict(pred)
