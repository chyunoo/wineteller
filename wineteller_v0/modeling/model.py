from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from wineteller.modeling.registry import load_model_knn
from .params import MAPPING
from .preprocessor import *
from operator import itemgetter
import copy

def word_embeddings(normalized_sentences) :
    model = Word2Vec(normalized_sentences,
                     vector_size=300,
                     min_count=1,
                     epochs=15)

    return model

def extract_descriptor(word):
    if word in MAPPING.keys() :
        descriptor_to_return = MAPPING[word]
        return descriptor_to_return


def descriptorize_reviews(data) :
    wine_reviews = list(data['description'])
    descriptorized_reviews = []
    for review in wine_reviews :
        normalized_review = normalizer(review)
        phrased_review = phraser(normalized_review)[normalized_review]
        descriptors_only = [extract_descriptor(word) for word in phrased_review]
        no_nones = [str(d) for d in descriptors_only if d is not None]
        descriptorized_review = ' '.join(no_nones)
        descriptorized_reviews.append(descriptorized_review)
    return descriptorized_reviews


def tfidf_weightings(descriptorized_reviews) :
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit(descriptorized_reviews)
    dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))

    return dict_of_tfidf_weightings

def vectorize_reviews(data, trained_model) :

    descriptorized_reviews = descriptorize_reviews(data)
    dict_of_tfidf_weightings = tfidf_weightings(descriptorized_reviews)

    wine_review_vectors = []
    for d in descriptorized_reviews:
        descriptor_count = 0
        weighted_review_terms = []
        terms = d.split(' ')
        for term in terms:
            if term in dict_of_tfidf_weightings.keys():
                tfidf_weighting = dict_of_tfidf_weightings[term]
                word_vector = trained_model.wv.get_vector(term).reshape(1, 300)
                weighted_word_vector = tfidf_weighting * word_vector
                weighted_review_terms.append(weighted_word_vector)
                descriptor_count += 1
            else:
                continue
        try:
            review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
        except:
            review_vector = []
        vector_and_count = [terms, review_vector, descriptor_count]
        wine_review_vectors.append(vector_and_count)

    return wine_review_vectors


def merge_review_vectors(wine_review_vectors, data) :

    new_data = copy.deepcopy(data)
    new_data['normalized_descriptors'] = list(map(itemgetter(0), wine_review_vectors))
    new_data['review_vector'] = list(map(itemgetter(1), wine_review_vectors))
    new_data['descriptor_count'] = list(map(itemgetter(2), wine_review_vectors))

    return new_data

def train_knn(df_mincount) :
    review_vectors_listed = list(df_mincount["review_vector"])
    knn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="cosine")
    knn_trained = knn.fit(review_vectors_listed)

    return knn_trained

def find_neighbors(occasion) :
    model = load_model_knn()
    avg_occasion = sum(occasion) / len(occasion)
    distance, indice = model.kneighbors([occasion], n_neighbors=10,return_distance=True)
    indice_list = indice[0].tolist()[:]

    return indice_list
