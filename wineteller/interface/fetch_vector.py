import numpy as np
import pandas as pd
import os
import sys

from wineteller.modeling.registry import load_model


def similar_word(word,n) :

   model = load_model()
   print(model.wv.index_to_key,)
   return model.wv.most_similar(positive=word, topn=int(n))


def fetch_vector(word) :

    # model = load_model()
    # list_ = []
    # for w in word :

    #     try :
    #         list_.append(model.wv[w])
    #     except KeyError :
    #         continue

    # return list_
    pass


if __name__ == '__main__':
    similar_word(*sys.argv[1:])
    fetch_vector(*sys.argv[1:])
