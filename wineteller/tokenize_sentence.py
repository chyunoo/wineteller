from os import lseek
from import_data import get_data
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize(data):
    reviews_list = list(data)
    reviews_list = [str(r) for r in reviews_list]
    sentences_tokenized=[]
    for review in reviews_list:
        sentences_tokenized.append(sent_tokenize(review))
    sentences_tokenized = [item for sublist in sentences_tokenized for item in sublist]
    return sentences_tokenized
