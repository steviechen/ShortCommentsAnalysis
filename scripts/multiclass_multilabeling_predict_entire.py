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

# proj_dir = '/data/part1/jupyter_project/StevieChen/OPS_Team/Logex/'
proj_dir = '/home/wbxbuilds/nps_analysis/data/'

# proj_dir = '/data/part1/jupyter_project/StevieChen/OPS_Team/Logex/'
    
def translateMultiLang(sentences):
    unknown_blob = TextBlob(sentences)
    if unknown_blob.detect_language() == 'en':
        return sentences
    else:
        return unknown_blob.translate(to='en').raw
    

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
    for sentence in sentences:
        stn = sentence.translate(str.maketrans(replacements)).strip()
        yield (stn.split(" "))
    
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
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

d = {
    "problems installing Webex":"installing_problems",
    "sign in multiple times":"sign_in_multiple_times",
    "wait a long time to join":"wait_longtime_to_join",
    "heard an echo":"heard_echo",
    "extra noise or distorted speech":"extra_noise_or_distorted_speech",
    "gap between responses":"gap_between_responses",
    "could not connect to the camera":"cannot_connect_camera",
    "could not start my video":"cannot_start_video",
    "experienced low-quality video":"low_quality_video",
    "video and audio were out of sync":"video_audio_out_of_sync",
    "audio stopped unexpectedly":"audio_stopped_unexpectedly",
    "could not share content":"cannot_share_content",
    "could not view shared content":"cannot_view_shared_content",
    "content was not in sync":"content_not_in_sync",
    "Others could not view content":"others_cannot_view_sharing_content"
    }

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
    for i in ast.literal_eval(ll):
        temp = temp + i + ' '
    return temp.rstrip()

def judge_value_with_nan(x):
    if np.isnan(x):
        return np.nan
    elif x > 8:
        return "PROMOTERS"
    elif x < 7:
        return "DETRACTORS"
    else: return "PASSIVES"
        
nps_all = pd.read_csv(proj_dir +'input/rawdata/nps_all.csv', encoding='latin-1')
nps_may_jun = pd.read_csv(proj_dir +'input/rawdata/nps_may_jun.csv', encoding='latin-1')
nps_jul_aug = pd.read_csv(proj_dir +'input/rawdata/nps_jul_aug.csv', encoding='latin-1')
nps_sep = pd.read_csv(proj_dir +'input/rawdata/nps_sep.csv', encoding='latin-1')
nps_oct = pd.read_csv(proj_dir +'input/rawdata/nps_oct.csv', encoding='latin-1')
nps_nov = pd.read_csv(proj_dir +'input/rawdata/nps_nov.csv', encoding='latin-1')
nps_dec = pd.read_csv(proj_dir +'input/rawdata/nps_dec.csv', encoding='latin-1')

nps_comments = nps_all.append(nps_may_jun).append(nps_jul_aug).append(nps_sep).\
    append(nps_oct).append(nps_nov).append(nps_dec).reset_index()

comments_temp = pd.concat([nps_comments[["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]], \
           pd.Series([np.all(x) for x in nps_comments[["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]].isna().values],\
           name="isnull").reset_index()["isnull"]], axis=1)
all_comments = comments_temp[comments_temp.iloc[:,5] == False]\
[["comments", "join_issue", "audio_issue", "video_issue", "sharing_issue"]].fillna(" ")
# nps_comments["concat_comments"] = all_comments["comments"]
nps_comments["concat_comments"] = all_comments["comments"] + \
    all_comments["join_issue"].apply(lambda x: ' ' + x) + \
    all_comments["audio_issue"].apply(lambda x: ' ' + x) + \
    all_comments["video_issue"].apply(lambda x: ' ' + x) + \
    all_comments["sharing_issue"].apply(lambda x: ' ' + x)

concatDF = nps_comments[["nps.date", "nps.siteid", "siteurl", "sitetype","nps.relversion",
  "version_info.version", "nps.domainname","version_info.usertype","version_info.os","version_info.apptype",
  "nps_score", "concat_comments", "join_issue","audio_issue","video_issue", "sharing_issue"]]
clean = concatDF[concatDF["nps.date"].isna()==False]
clean["nps_score"][clean["nps_score"]==-1] = np.nan

clean[["datetime", "siteid", "siteurl", "sitetype","relversion","cliversion", 'domainname', 'nps_score']] = \
nps_comments[["nps.date", "nps.siteid", "siteurl", "sitetype","nps.relversion",
  "version_info.version", "nps.domainname", "nps_score"]]
clean["other_issue"] = pd.Series(np.nan)
clean["other_issue"] = clean["other_issue"].fillna(" ")
clean[["join_issue", "audio_issue", "video_issue", "sharing_issue", "concat_comments"]] = \
concatDF[["join_issue", "audio_issue", "video_issue","sharing_issue", "concat_comments"]].fillna(" ")

clean["appended_comments"] = clean["concat_comments"].apply(lambda x: find_replace_multi(x, d))
clean["vc_category"] = clean["nps_score"].\
apply(judge_value_with_nan)

clean = clean[["datetime", "siteid", "siteurl", "sitetype","relversion","cliversion",\
  'domainname', 'nps_score', "concat_comments", "join_issue","audio_issue","video_issue",\
  "sharing_issue", "other_issue", 'appended_comments', 'vc_category']]

predict_df = clean[clean["vc_category"] == 'DETRACTORS'][clean["concat_comments"].apply(lambda x: x != ' ')].reset_index()


# root_dir = "/data/part1/jupyter_project/StevieChen/OPS_Team/Logex/"

max_words = 5000
max_len = 200

embed_size = 300

checkpoint_path = proj_dir + "output/ckp/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
classifier_path = proj_dir + "output/model/multi_labeling_classfier.h5"


with open(proj_dir + 'output/model/tokenizer.pkl', 'rb') as pickle_file:
    tokenizer_pkl = pkl.load(pickle_file)

    
x_predict = predict_df["concat_comments"].str.lower()
x_predict = tokenizer_pkl.texts_to_sequences(x_predict)
x_predict = tf.keras.preprocessing.sequence.pad_sequences(x_predict, maxlen=max_len)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model = tf.keras.models.load_model(classifier_path)

model.load_weights(latest)

def get_first(ll):
    if len(ll) > 0:
        return ll[0]
    else:
        return ''

result = []
for elem in x_predict:
    temp = model.predict(np.expand_dims(elem, 0))
    result.append(temp)
    
labels = pd.read_csv(proj_dir + 'output/model/labels.csv')
tt = []
for i in result:
    temp = [elem for elem in labels[i[0] > 0.5]["lable_name"]\
            if elem in ['UI issue','app/product not work properly','general issue during meeting']]
    tt.append(temp)
    
predict_df = pd.concat([predict_df[['index', 'datetime', 'siteid', 'siteurl', 'sitetype', 'relversion', 'cliversion',
       'domainname', 'nps_score', 'concat_comments', 'join_issue',
       'audio_issue', 'video_issue', 'sharing_issue', 'appended_comments', 'vc_category']], 
        pd.Series(tt, name="other_issue").apply(get_first)], axis = 1)
clean = pd.concat([clean[['datetime', 'siteid', 'siteurl', 'sitetype', 'relversion', 'cliversion',
       'domainname', 'nps_score', 'concat_comments', 'join_issue',
       'audio_issue', 'video_issue', 'sharing_issue', 'appended_comments', 'vc_category']],\
        predict_df.set_index(predict_df["index"])["other_issue"]], axis=1)
clean["other_issue"] = clean["other_issue"].fillna(" ")
data_words = list(sent_to_words(clean["concat_comments"].values))
bigram = gensim.models.Phrases(data_words, min_count=1)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words])

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

stop_words = stopwords.words('english')
stop_words.extend(
    ['from', 'webex', 'cisco', 'org', 'com', 'id', 'please', 'hello', 'hi', 'greeting', 'greetings', 'today'])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words, stop_words)


# Form Bigrams
data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

data_words_trigrams = make_trigrams(trigram_mod, bigram_mod, data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

#     Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(nlp, data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# Create Corpus
contents = pd.Series(data_words_trigrams, name="comm_wl")
clean["comments"] = clean["concat_comments"]
final_df = pd.concat([clean[["datetime", "siteid", "siteurl", "sitetype", "relversion", \
    "cliversion", "domainname", "nps_score", "comments", "join_issue", "audio_issue", "video_issue", \
    "sharing_issue", "other_issue", "vc_category"]], contents], axis=1)

print("final datashape is " + str(final_df.shape))
final_df.to_sql('STAPUSER.STAP_NPS_COMMENTS_ALL', engine, if_exists='append', chunksize = 100, index=False)
