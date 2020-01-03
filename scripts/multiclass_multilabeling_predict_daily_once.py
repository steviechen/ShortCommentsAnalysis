import tensorflow as tf

import numpy as np
import pandas as pd
import re

from ast import literal_eval

import os
import gensim
from gensim.utils import simple_preprocess
import pickle as pkl
#nlp
import spacy
import nltk
from nltk.corpus import stopwords

# proj_dir = '/data/part1/jupyter_project/StevieChen/OPS_Team/Logex/'
proj_dir = '/home/wbxbuilds/nps_analysis/data/'

def sent_to_words(sentences):
    replacements = {
        "\'": " ",
        ".": " ",
        ",": " ",
        '"': ' ',
        "{": " ",
        "}": " ",
        "[": " ",
        "]": " ",
        "(": " ",
        ")": " ",
        "?": " ",
        "#": " ",
        ":": " ",
        "~": " ",
        "!": " ",
        "$": " ",
    }
    #     common_out = generate_words_from_file('/Users/steviechen1982/PycharmProjects/Logex/rawdata/common_words')
    for sentence in sentences:
        stn = sentence.translate(str.maketrans(replacements)).strip()
        yield (stn.split(" "))

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(trigram_mod, bigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def find_replace_multi(string, dictionary):
    for item in dictionary.keys():
        # sub item for item's paired value in string
        string = re.sub(item, dictionary[item], string)
    return string


def get_labels(ls, dictionary):
    res = []
    for elem in ls:
        for item in dictionary.values():
            if elem == item:
                res.append(item)
    return res


def conc_list(ll):
    temp = ''
    for i in literal_eval(ll):
        temp = temp + i + ' '
    return temp.rstrip()


import datetime as dt
from dateutil.relativedelta import relativedelta
from sqlalchemy.engine import create_engine
from sqlalchemy import types

DIALECT = 'oracle'
SQL_DRIVER = 'cx_oracle'
USERNAME = 'stapuser' #enter your username
PASSWORD = 'se#0stpdb' #enter your password
HOST = '10.252.8.134' #enter the oracle db host url
PORT = 1909 # enter the oracle port number
SID='stapdb1'
# SERVICE = 'your_oracle_service_name' # enter the oracle db service name
ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/' + SID

engine = create_engine(ENGINE_PATH_WIN_AUTH)

final_df = pd.read_csv(proj_dir + 'output/df/dec_once.csv', encoding='latin-1')


final_df["comments"] = final_df["comments"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["join_issue"] = final_df["join_issue"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["audio_issue"] = final_df["audio_issue"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["video_issue"] = final_df["video_issue"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["sharing_issue"] = final_df["sharing_issue"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["other_issue"] = final_df["other_issue"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
final_df["comm_wl"] = final_df["comm_wl"].apply(lambda x: str(x).encode('utf-8', 'ignore'))
print("final datashape is " + str(final_df.shape))
# final_df.to_csv(proj_dir + 'output/df/incre_once.csv')
final_df.to_sql('STAPUSER.STAP_NPS_COMMENTS_ALL', engine, if_exists='append', chunksize = 100, index=False)