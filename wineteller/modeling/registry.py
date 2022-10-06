import os
import time
from gensim.models import Word2Vec
import glob
from colorama import Fore, Style



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

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]

    model = Word2Vec.load(model_path)

    print(f"\n✅ model loaded from {model_path}")

    return model
