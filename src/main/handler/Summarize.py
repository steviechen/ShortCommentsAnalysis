from math import isnan

import json
from tornado import gen
from tornado.concurrent import run_on_executor

from gensim.models.word2vec import Word2Vec
from gensim.models import Doc2Vec
from gensim.models import KeyedVectors
from gensim import corpora
import numpy as np
import pandas as pd
import re
import codecs

import subprocess
from collections import namedtuple
from collections import defaultdict

import datetime as dt
import os
import sys

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk import tokenize

import time
from itertools import chain
import json
from re import sub
from os.path import isfile

import gensim.downloader as api
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk import download

from multiprocessing import cpu_count

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
import pickle as pkl

from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
import numpy as np
from sklearn.model_selection import KFold
from wmd import WMD
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_distances

import time

from src.main.domain.vo.TicketIndex import TicketIndex, Desc
from src.main.repository.oracle.NPSOracleEngine import NPSOracleEngine
from src.main.utils.ModelingUtil import ModelingUtil
from src.main.utils.TextProcessUtil import TextProcessUtil

def log(stri):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(str(now) + ' ' + str(stri))

class SummarizeHandler():
    def __init__(self):
        super().__init__()
        self.tpUtil = TextProcessUtil()
        self.modelUtil = ModelingUtil()
        self.oracleEng = NPSOracleEngine()
        self.currdate = dt.datetime.now().strftime('%Y%m%d')

    # def getRecent3mDF(self):
    #     # nps_sep = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_sep.csv', encoding='latin-1')
    #     nps_oct = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_oct.csv', encoding='latin-1')
    #     nps_nov = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_nov.csv', encoding='latin-1')
    #     nps_dec = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_dec.csv', encoding='latin-1')
    #
    #     nps_comments = nps_oct.append(nps_nov).append(nps_dec).reset_index()
    #     return nps_comments

    def getRecentNMonthDF(self, month):
        nps_comments = self.oracleEng.queryNMonth(month)
        return nps_comments

    def generateResult(self, current_ldamodel, corpus, data_words, texts, origin_df):
        df_topic_sents_keywords = self.tpUtil.format_topics_sentences(ldamodel=current_ldamodel, corpus=corpus, texts=data_words)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        result = pd.concat([origin_df.reset_index(), df_dominant_topic.reset_index(), pd.Series(texts, name="words list")], axis=1)
        return result

    def summWeekly(self, senti_type, month=3):
        nps_comments = self.getRecentNMonthDF(month)
        clean = self.tpUtil.preprocessDF(nps_comments)
        data_words, appended = self.tpUtil.generateWordsList(clean, senti_type)
        texts, id2word, corpus = self.tpUtil.generateCorpus(data_words, appended)
        current_ldamodel = self.modelUtil.loadModel(senti_type)
        result = self.generateResult(current_ldamodel, corpus, data_words, texts,
                                     clean[clean["vc_category"] == senti_type])
        result.to_csv('/Users/steviechen1982/Documents/NPS/output/df/weekly/NPS_%s_dominant_topic_%s.csv' %(senti_type, self.currdate), encoding='utf8', index=None)
        return 'Summarization process has been finished.'