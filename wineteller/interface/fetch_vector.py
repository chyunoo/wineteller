import numpy as np
import pandas as pd
import os
import sys

from wineteller.modeling.registry import load_model


def similar_word(word,n) :

   model = load_model()
   print(model.wv.index_to_key,)
   return model.wv.most_similar(positive=word, topn=int(n))


def fetch_vector(words) :
    """
    fetch vector
    """
    model = load_model()
    list_ = []
    for w in words :
        if w in model.wv.index_to_key :
            list_.append(model.wv[w])
            #print(f'{w} not found')

    return list_


if __name__ == '__main__':
    similar_word(*sys.argv[1:])
    fetch_vector(*sys.argv[1:])
