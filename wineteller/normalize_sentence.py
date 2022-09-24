from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

stop_words = set(stopwords.words('english'))

punctuation_table = str.maketrans({key: None for key in string.punctuation})
sno = SnowballStemmer('english')

#This function normalized sentence from sentence already tokenized.
def normalize_text(sentence_tokenized):
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
