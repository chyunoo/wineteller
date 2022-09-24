from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from os import lseek
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.phrases import Phrases, Phraser


def tokenize(data):
    reviews_list = list(data["description"])
    reviews_list = [str(r) for r in reviews_list]

    sentences_tokenized=[]
    for review in reviews_list:
        sentences_tokenized.append(sent_tokenize(review))
    sentences_tokenized = [item for sublist in sentences_tokenized for item in sublist]

    return sentences_tokenized



def stop_words_punc() :
     #return stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans({key: None for key in string.punctuation})
    sno = SnowballStemmer('english')
    return stop_words, punctuation_table, sno


def normalizer(sentence_tokenized):
    #This function normalized sentence from sentence already tokenized.
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

def normalize_text(sentences_tokenized) :
    normalized_sentences = []
    for s in sentences_tokenized :
        normalized_text = normalizer(s)
        normalized_sentences.append(normalized_text)

    return normalized_sentences

def ngrams(normalized_sentences) :
    phrases = Phrases(normalized_sentences)
    phrases = Phrases(phrases[normalized_sentences])
    ngrams = Phraser(phrases)

    return ngrams

def phrase_text(normalized_sentences) :
    phrased_sentences = []
    for s in normalized_sentences:
        phrased_sentence = ngrams[s]
        phrased_sentences.append(phrased_sentence)

    full_list_words = [item for sublist in phrased_sentences for item in sublist]

    return full_list_words

def filtered_mapping(mp, main_descriptors = ['body',
                                            'complexity',
                                            'finish',
                                            'alcohol',
                                            'sweetness']) :

    filtered_descriptor_mapping = mp[mp['level_1'].isin(main_descriptors)]

    descriptors_list = list(filtered_descriptor_mapping["level_1"].unique())
    print(f"\n🍷 used {len(descriptors_list)} descriptors : {descriptors_list}")

    return filtered_descriptor_mapping


def descriptor_mapping(word, fmp):
    if word in list(fmp.index):
        normalized_word = fmp['level_3'][word]
        return normalized_word
    else:
        return ""

def return_descriptor_mapping(phrased_sentences):
    normalized_sentences = []
    for sent in phrased_sentences:
        normalized_sentence = []
        for word in sent:
            normalized_word = descriptor_mapping(word)
            if normalized_word != "" :
                normalized_sentence.append(str(normalized_word))
            else :
                pass
    normalized_sentences.append(normalized_sentence)
    return normalized_sentences
