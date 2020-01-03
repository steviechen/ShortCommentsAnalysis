############################ 训练 ############################
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
# reload(sys)
# sys.setdefaultencoding('utf8')

#misc
import gc
import time
import warnings

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
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk import download

# from multiprocessing import cpu_counts

import pickle as pkl

import time

from src.main.utils.TextProcessUtil import TextProcessUtil
from src.main.utils.ModelingUtil import ModelingUtil

# currdate = dt.datetime.now().strftime('%F')

def log(stri):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(str(now) + ' ' + str(stri))


class TrainingHandler(object):
    def __init__(self):
        super().__init__()
        self.tpUtil = TextProcessUtil()
        self.modelUtil = ModelingUtil()
        self.currdate = dt.datetime.now().strftime('%Y%m%d')

    def getEntireDF(self):
        nps_all = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_all.csv', encoding='latin-1')
        nps_may_jun = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_may_jun.csv',
                                  encoding='latin-1')
        nps_jul_aug = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_jul_aug.csv',
                                  encoding='latin-1')
        nps_sep = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_sep.csv', encoding='latin-1')
        nps_oct = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_oct.csv', encoding='latin-1')
        nps_nov = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_nov.csv', encoding='latin-1')
        nps_dec = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/nps_dec.csv', encoding='latin-1')

        nps_comments = nps_all.append(nps_may_jun).append(nps_jul_aug).append(nps_sep).\
            append(nps_oct).append(nps_nov).append(nps_dec).reset_index()
        nps_comments.columns = ['index', 'datetime', 'siteid', 'siteurl', 'sitetype',
                                'relversion', 'cliversion', 'domainname',
                                'version_info.usertype', 'version_info.os', 'version_info.apptype',
                                'nps_score', 'stars_score', 'comments', 'join_issue', 'audio_issue',
                                'video_issue', 'sharing_issue']
        return nps_comments

    def generateResult(self, current_ldamodel, corpus, data_words, texts, origin_df):
        df_topic_sents_keywords = self.tpUtil.format_topics_sentences(ldamodel=current_ldamodel, corpus=corpus, texts=data_words)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        result = pd.concat([origin_df.reset_index(), df_dominant_topic.reset_index(), pd.Series(texts, name="words list")], axis=1)
        return result

    def trainingProcess(self, senti_type, topic_num):
        nps_comments = self.getEntireDF()
        clean = self.tpUtil.preprocessDF(nps_comments)
        data_words, appended = self.tpUtil.generateWordsList(clean, senti_type)
        texts, id2word, corpus = self.tpUtil.generateCorpus(data_words, appended)
        current_ldamodel = self.modelUtil.trainSaveModel(senti_type, topic_num, corpus, id2word)
        result = self.generateResult(current_ldamodel, corpus, data_words, texts, clean[clean["vc_category"] == senti_type])
        result.to_csv('/Users/steviechen1982/Documents/NPS/output/df/NPS_%s_dominant_topic_%s.csv' %(senti_type, self.currdate), encoding='utf8', index=None)
        return 'Training process has been finished.'