from math import isnan

import string
import json
import numpy as np
import pandas as pd
import re
from re import sub
import codecs
from itertools import chain
from multiprocessing import cpu_count

from ast import literal_eval
# from tornado import gen
# from tornado.concurrent import run_on_executor

import subprocess
from collections import namedtuple
from collections import defaultdict

import datetime as dt
import os
from os.path import isfile
import sys
import gc
import time
import warnings

import gensim
import spacy

import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models import KeyedVectors
from gensim import corpora
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.matutils import softcossim
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
import pickle as pkl
from nltk.corpus import stopwords



class TextProcessUtil(object):
    # def __init__(self):
    #     self.host = ConfigurationUtil.get(self.config_key, 'host')
    #     self.port = ConfigurationUtil.get(self.config_key, 'port')
    #     self.db = ConfigurationUtil.get(self.config_key, 'db')
    #     self.redis_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.db)
    #     self.redis_client = redis.Redis(connection_pool=self.redis_pool)

    # def sent_to_words(sentences):
    #     '''Tokenize words and Clean-up text'''
    #     for sentence in sentences:
    #         yield (simple_preprocess(str(sentence), deacc=True))
    #         yield (simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def sent_to_words(self, sentences):
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


    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts, stop_words):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


    # Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
    def make_bigrams(self, bigram_mod, texts):
        return [bigram_mod[doc] for doc in texts]


    def make_trigrams(self, trigram_mod, bigram_mod, texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]


    def lemmatization(self, nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    def format_topics_sentences_mallet(self, ldamodel: LdaMallet, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num, topn=8)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)


    def format_topics_sentences(self, ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    d = {
        "problems installing Webex": "installing_problems",
        "sign in multiple times": "sign_in_multiple_times",
        "wait a long time to join": "wait_longtime_to_join",
        "heard an echo": "heard_echo",
        "extra noise or distorted speech": "extra_noise_or_distorted_speech",
        "gap between responses": "gap_between_responses",
        "could not connect to the camera": "cannot_connect_camera",
        "could not start my video": "cannot_start_video",
        "experienced low-quality video": "low_quality_video",
        "video and audio were out of sync": "video_audio_out_of_sync",
        "audio stopped unexpectedly": "audio_stopped_unexpectedly",
        "could not share content": "cannot_share_content",
        "could not view shared content": "cannot_view_shared_content",
        "content was not in sync": "content_not_in_sync",
        "Others could not view content": "others_cannot_view_sharing_content"
    }


    def find_replace_multi(self, string, dictionary):
        for item in dictionary.keys():
            # sub item for item's paired value in string
            string = re.sub(item, dictionary[item], string)
        return string


    def get_labels(self, ls, dictionary):
        res = []
        for elem in ls:
            for item in dictionary.values():
                if elem == item:
                    res.append(item)
        return res

    def preprocessDF(self, nps_comments):
        comments_temp = pd.concat(
            [nps_comments[["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]],
             pd.Series([np.all(x) for x in nps_comments[
                 ["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]].isna().values],
                       name="isnull").reset_index()["isnull"]], axis=1)
        all_comments = comments_temp[comments_temp.iloc[:, 5] == False][
            ["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]].fillna(" ")

        nps_comments["concat_comments"] = all_comments["comments"] + \
                                          all_comments["join_issue"].apply(lambda x: ' ' + x) + \
                                          all_comments["audio_issue"].apply(lambda x: ' ' + x) + \
                                          all_comments["video_issue"].apply(lambda x: ' ' + x) + \
                                          all_comments["sharing_issue"].apply(lambda x: ' ' + x)

        concatDF = nps_comments[
            ["datetime", "siteid", "siteurl", "sitetype", "relversion", 'cliversion', 'domainname',
             "nps_score", "concat_comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]]
        clean = concatDF[concatDF["concat_comments"].isna() == False]
        clean["appended_comments"] = clean["concat_comments"].apply(
            lambda x: self.find_replace_multi(x, self.d))

        clean["vc_category"] = clean["nps_score"].apply(lambda x: "PROMOTERS" if x > 8 else "DETRACTORS" if x < 7 else "PASSIVES")
        return clean

    def generateWordsList(self, clean, senti_type="DETRACTORS"):
        data = clean["concat_comments"][clean["vc_category"] == senti_type].values

        appended = clean["appended_comments"][clean["vc_category"] == senti_type].values
        return list(self.sent_to_words(data)), list(self.sent_to_words(appended))

    def generateCorpus(self, data_words, appended):
        bigram = gensim.models.Phrases(data_words, min_count=1)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words])

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        stop_words = stopwords.words('english')
        stop_words.extend(
            ['from', 'webex', 'cisco', 'org', 'com', 'id', 'please', 'hello', 'hi', 'greeting', 'greetings', 'today'])

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words, stop_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(bigram_mod, data_words_nostops)

        data_words_trigrams = self.make_trigrams(trigram_mod, bigram_mod, data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        #     Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(nlp, data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Corpus
        texts = data_words_trigrams + pd.Series(appended).apply(lambda x: self.get_labels(x, self.d))

        # Create Dictionary
        id2word = corpora.Dictionary(texts)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        return texts, id2word, corpus

    def generateCleanSentence(self, se):
        data = se.values

        data_words = list(self.sent_to_words(data))

        stop_words = stopwords.words('english')
        stop_words.extend(
            ['from', 'webex', 'cisco', 'org', 'com', 'id', 'please', 'hello', 'hi', 'greeting', 'greetings', 'today'])

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words, stop_words)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        #     Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(nlp, data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        return pd.Series([' '.join(line) for line in data_lemmatized], name="concat_comments")