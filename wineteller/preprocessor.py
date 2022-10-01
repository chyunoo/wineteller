from wineteller.preprocessing import *
from wineteller.import_data import *

import numpy as np
import pandas as pd
import pickle
import time

from colorama import Fore, Style

#Tracker
from .utils import simple_time_and_memory_tracker
@simple_time_and_memory_tracker

def preprocess_text(data : pd.DataFrame) -> list :
    """
    Preprocess wine descriptions (tokenize, normalize, phrase, map) and
    concatenate in a list of sentences ready to be trained in Word2Vec
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nPreprocess data ..." + Style.RESET_ALL)

    print(f"\n tokenizing {len(data)} 🍷 descriptions ...")
    tokenized_sentences = tokenize_text(data)
    print(f"✔️ text tokenized : {len(tokenized_sentences)}")

    print(f"normalizing {len(tokenized_sentences)} sentences ...")
    normalized_sentences = normalize_text(tokenized_sentences)
    print(f"✔️ text normalized : {len(normalized_sentences)}")

    print(f"phrasing {len(normalized_sentences)} sentences ...")
    phrased_sentences = phrase_text(normalized_sentences)
    print(f"✔️ text phrased : {len(phrased_sentences)}")

    mp = get_data("descriptor_mapping")
    fmp = filtered_descriptor_mapping(mp)
    print(f"✔️ imported mapping : {len(fmp)} raw descriptors")

    print(f"mapping {len(phrased_sentences)} sentences ...")
    preprocessed_text = mapping_text(phrased_sentences)
    print(f"✔️ text mapped : {len(preprocessed_text)}")

    """
    allow below if you want to save locally preprocessed_text
    """
    # if len(preprocessed_text) > 0 :
    #     file_path= os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                             "raw_data", timestamp + ".preprocessed")
    #     with open(file_path, "wb") as file:
    #         pickle.dump(preprocessed_text, file)
    #         print(f"saved locally at {file_path}")

    print("\n ✅ text processed")

    return preprocessed_text
