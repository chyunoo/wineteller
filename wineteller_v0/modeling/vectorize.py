import pandas as pd
import numpy as np
import os
from gensim.models import word2vec
from colorama import Fore, Style
from wineteller.interface.fetch_vector import fetch_vector
from wineteller.modeling.import_data import get_data
from sklearn.neighbors import NearestNeighbors
from wineteller.modeling.params import SPECIFIC_MP
import nltk

def clean_survey(survey) :
    data = survey.iloc[2:]
    print(data.shape)

    test_fr = data.iloc[:,:87]
    test_fr =test_fr.drop(columns=["What is your language 🗣 ? ","Timestamp"])
    test_en = data.iloc[:,87:]
    new_cols = {x: y for x, y in zip(test_fr.columns, test_en.columns)}
    test = test_en.append(test_fr.rename(columns=new_cols))
    test=test.set_axis(test.columns.str.replace("Length","Finish"),axis=1)

    for t in test.keys():
        test[t]=test[t].replace("Moyen","Medium")
        test[t]=test[t].replace("Faible","Low")
        test[t]=test[t].replace("Fort","High")

    high_low=["Alcohol","Complexity"]
    for hl in high_low:
        for t in test.keys():
            if hl in t:
                test[t]=test[t].replace("Medium","NaN")
            else:
                continue

    no_low=["Finish"]
    for nl in no_low:
        for t in test.keys():
            if nl in t:
                test[t]=test[t].replace("Low","NaN")
            else:
                continue


    return test

def useful_lists() :

    #### changed after work to work
    #### changed get drunk to drunk
    word = ["gift","family","friends","date","dinner","lunch","barbecue","outdoor","summer","winter","restaurant","work","colleagues","drunk","home","fancy","birthday"]
    cara =  ["Complexity", "Body","Sweetness","Alcohol"]
    non_aroma={"Body":["full_bodied","light_bodied","medium_bodied"],"Alcohol":["high_alcohol","low_alcohol"],"Sweetness":["very_sweet","dry","sweet"],"Complexity":["high_complexity","low_complexity"],"Finish":["medium_length_finish","long_finish"]}

    return word, cara, non_aroma

def vectorize_survey(test) :

    breakdown={}
    for column in test.keys():
        breakdown[column]=test[column][test[column].isin(["Low","Medium","High"])==True].value_counts(normalize=True).sort_index()

    word, cara, non_aroma = useful_lists()

    cool_matrix={}
    for keys in breakdown.keys():
        for carac in cara :
            for words in word:
                if (words in keys) and (carac in keys):
                    cool_matrix[words +" " + carac.lower()]=breakdown[keys]

    corpus = {}
    for k in cara:
        for i in non_aroma[k]:
            #print(k+""+i)
            specific = SPECIFIC_MP[i]
            data_=np.array(fetch_vector(list(specific))).mean(axis=0)
            try:
                corpus[k.lower()]=np.column_stack((data_,corpus[k.lower()]))
                #corpus[k]=data_
            except KeyError:
                if data_.shape==(300,):
                    corpus[k.lower()]=data_
                else:
                    continue

    final={}
    for w in word:
        for c in cara:
            try:
                matrix_=np.dot(corpus[c.lower()],np.array(cool_matrix[w + " " + c.lower()]))
                if w in final.keys():
                    final[w]=np.column_stack((matrix_,final[w]))
                else:
                    final[w]=matrix_
            except ValueError:
                pass
        final[w]=np.mean(final[w],axis=1)

    return final
