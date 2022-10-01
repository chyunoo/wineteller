import numpy as np
import pandas as pd
import os
import sys

from wineteller.registry import load_model


def similar_word(word="light_bodied", n=3) :

    model = load_model()
    print(model.wv.index_to_key,)
    print(model.wv.most_similar(positive=word, topn=int(n)))


if __name__ == '__main__':
    similar_word(*sys.argv[1:])
