from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from os import lseek
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.phrases import Phrases, Phraser
import pandas as pd

from .params import MAPPING


def tokenize_text(data : pd.DataFrame) -> list :
    """
    split/tokenize wine descriptions into sentences, load wine review
    one by one then concatenate into list
    """
    reviews_list = list(data["description"])
    reviews_list = [str(r) for r in reviews_list]

    sentences_tokenized=[]
    for review in reviews_list:
        sentences_tokenized.append(sent_tokenize(review))
    sentences_tokenized = [item for sublist in sentences_tokenized for item in sublist]

    return sentences_tokenized


def stop_words_punc() :
    """
    retrieve stop_words for normalizer
    """
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans({key: None for key in string.punctuation})
    sno = SnowballStemmer('english')
    return stop_words, punctuation_table, sno


def normalizer(sentence_tokenized) :
    """
    define function to remove punctuation and stop-words
    """
    stop_words, punctuation_table, sno = stop_words_punc()

    try:
        word_list = word_tokenize(sentence_tokenized)
        normalized_sentence = []
        for w in word_list:
            try:
                w = str(w)
                lower_case_word = str.lower(w)
                stemmed_word = sno.stem(lower_case_word)
                no_punctuation = stemmed_word.translate(punctuation_table)
                if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                    normalized_sentence.append(no_punctuation)
            except:
                continue
        return normalized_sentence
    except:
        return ''

def normalize_text(sentences_tokenized : list) -> list :
    """
    apply normalizer function on sentences
    """
    normalized_sentences = []
    for s in sentences_tokenized :
        normalized_text = normalizer(s)
        normalized_sentences.append(normalized_text)

    return normalized_sentences

def phraser(normalized_sentences) :
    """
    define function to retrieve ngrams from sentences
    """
    phrases = Phrases(normalized_sentences)
    phrases = Phrases(phrases[normalized_sentences])
    ngrams = Phraser(phrases)

    return ngrams

def phrase_text(normalized_sentences : list) -> list :
    """
    apply phraser on sentences
    """
    phrased_sentences = []
    ngrams = phraser(normalized_sentences)

    for s in normalized_sentences:
        phrased_sentence = ngrams[s]
        phrased_sentences.append(phrased_sentence)

    return phrased_sentences

def filtered_descriptor_mapping(mp : pd.DataFrame, main_descriptors = ['body',
                                            'complexity',
                                            'finish',
                                            'alcohol',
                                            'sweetness']) -> dict :
    """
    filter wine descriptors mapping with list of main descriptors and save mapping
    in dict format (raw_descriptor : descriptor_level_3)
    """

    filtered_descriptor_mapping = mp[mp['level_1'].isin(main_descriptors)]
    descriptors_list = list(filtered_descriptor_mapping["level_1"].unique())
    print(f"using {len(descriptors_list)} 🍷 descriptors : {descriptors_list}")
    filtered_descriptor_mapping.set_index("raw descriptor", inplace=True)

    mapping = {}
    for raw_descriptor, descriptor in zip(filtered_descriptor_mapping.index,
                              list(filtered_descriptor_mapping["level_3"])) :
        mapping[raw_descriptor] = descriptor
    return mapping

    #### code needed to automatically export in params module ####

def mapper(word):
    """
    define function to map words with a filtered descriptor mapping (stored in
    params module)
    """

    if word in MAPPING.keys():
        normalized_word = MAPPING[word]
        return normalized_word
    else:
        return ""

def mapping_text(phrased_sentences : list) -> list:
    """
    apply mapper on sentences
    """

    mapped_sentences = []
    for sent in phrased_sentences:
        mapped_sentence = []
        for word in sent:
            mapped_word = mapper(word)
            if mapped_word != "" :
                mapped_sentence.append(str(mapped_word))
            else :
                pass
        mapped_sentences.append(mapped_sentence)

    return mapped_sentences
