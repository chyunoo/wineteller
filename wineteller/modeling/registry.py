import os
import time
from gensim.models import Word2Vec
import glob
from colorama import Fore, Style
from sklearn.neighbors import NearestNeighbors
import pickle

def save_model(model: Word2Vec = None) -> None:
    """
    persist trained model
    """
    if model is not None :
        model_path= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "raw_data","trained_model", "trained_w2v.model")
        print(f"\n✅ model saved locally at {model_path}")
        model.save(model_path)

    return None

def load_model() -> Word2Vec :
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk.." + Style.RESET_ALL)

    model_directory= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "raw_data", "trained_model")

    results = glob.glob(f"{model_directory}/*w2v*")
    if not results:
        return None

    model_path = sorted(results)[-1]

    model = Word2Vec.load(model_path)

    print(f"\n✅ model loaded from {model_path}")

    return model

def save_model_knn(model: NearestNeighbors = None) -> None:
    """
    persist trained model
    """
    if model is not None :
        model_path= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "raw_data","trained_model", "trained_knn.model")

        with open(model_path, "wb") as file:
            pickle.dump(model, file)
            print(f"\n✅ knn model saved locally at {model_path}")
            file.close()

    return None


def load_model_knn() -> NearestNeighbors :
    """
    load the latest saved model, return None if no model found
    """
    model_directory= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "raw_data", "trained_model")

    results = glob.glob(f"{model_directory}/*knn*")
    if not results:
        return None

    model_path = sorted(results)[-1]

    with open(model_path, "rb") as file:
        model = pickle.load(file)
        print(f"\n✅ model loaded from {model_path}")

        return model
