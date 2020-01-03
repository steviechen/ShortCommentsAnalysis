import gensim
from gensim.models import LdaModel
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import pickle as pkl

import os

from src.main.utils.TextProcessUtil import TextProcessUtil


class ModelingUtil(object):
    def __init__(self):
        self.max_words = 5000
        self.max_len = 200
        self.currdate = dt.datetime.now().strftime('%Y%m%d')
        self.tputil = TextProcessUtil()

    def trainSaveModel(self, senti_type, topic_num, corpus, id2word):
        mallet_path = '/Users/steviechen1982/Documents/NPS/input/mallet-2.0.8/bin/mallet'  # update this path
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=topic_num,
                                                     id2word=id2word, iterations=3000, random_seed=25,
                                                     alpha=60, topic_threshold=0.1)
        current_ldamodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
        current_ldamodel.save('/Users/steviechen1982/Documents/NPS/output/model/nps_lda_model_%s' % (senti_type))
        return current_ldamodel

    def loadModel(self, senti_type):
        new_ldamodels = LdaModel.load('/Users/steviechen1982/Documents/NPS/output/model/nps_lda_model_%s' % (senti_type))
        # new_ldamodels.update(new_corpus)
        return new_ldamodels

    def trainMultiLabel(self, rawdata_cla, glove_path, embed_size, callbacks, batch_size=128):
        data_cla = self.getDFWithDummy(rawdata_cla)
        pd.Series(data_cla.columns[7:], name="lable_name").\
            to_csv('/Users/steviechen1982/Documents/NPS/output/model/labels.csv', index=None, header=["lable_name"])

        # x_train = rawdata_cla["concat_comments"].str.lower()
        x_train = self.tputil.generateCleanSentence(rawdata_cla["concat_comments"])

        y_train = data_cla[data_cla.columns[7:]].values

        output_size = len(data_cla.columns[7:])
        # print(output_size)

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_words, lower=True)
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.max_len)

        embedding_matrix = self.embedding(glove_path, tokenizer, self.max_words, embed_size)

        input = tf.keras.layers.Input(shape=(self.max_len,))

        x = tf.keras.layers.Embedding(self.max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)

        model = self.buildNN(x, input, output_size=output_size)

        model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=100, callbacks=callbacks, verbose=1)

        return model

    def predictMultiLabelSingle(self, sentence, model, threshhold=0.2):
        # x_predict = pd.Series(sentence).str.lower()
        x_predict = self.tputil.generateCleanSentence(pd.Series(sentence, name="concat_comments"))
        # tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_words, lower=True)
        # tokenizer.fit_on_texts(x_predict)
        with open('/Users/steviechen1982/Documents/NPS/output/model/tokenizer.pkl', 'rb') as pickle_file:
            tokenizer_pkl = pkl.load(pickle_file)

        x_predict = tokenizer_pkl.texts_to_sequences(x_predict)
        x_predict = tf.keras.preprocessing.sequence.pad_sequences(x_predict, maxlen=self.max_len)
        predictions = model.predict(np.expand_dims(x_predict[0], 0))
        labels = pd.read_csv('/Users/steviechen1982/Documents/NPS/output/model/labels.csv')
        # return labels[predictions[0] > threshhold].to_json()
        return pd.concat([labels[predictions[0] > threshhold].reset_index(), pd.Series(predictions[0][predictions[0] > threshhold], name="prob")],
                    axis=1).to_json(orient='records')


    def getDFWithDummy(self, rawdata_cla):
        return pd.concat(
            [rawdata_cla[["siteid", "cliversion", "domainname", "nps_score", "concat_comments", "Text", "words list"]],
             pd.get_dummies(rawdata_cla["join_issue"]), pd.get_dummies(rawdata_cla["audio_issue"]),
             pd.get_dummies(rawdata_cla["video_issue"]), pd.get_dummies(rawdata_cla["sharing_issue"]),
             pd.get_dummies(rawdata_cla["other_issue"])], axis=1)

    def embedding(self, glove_path, tokenizer, max_words, embed_size):
        embeddings_index = {}

        with open(glove_path, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embed

        word_index = tokenizer.word_index

        num_words = min(max_words, len(word_index) + 1)

        embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')

        for word, i in word_index.items():

            if i >= max_words:
                continue

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def buildNN(self, x, input: tf.keras.Input, output_size):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                              recurrent_dropout=0.1))(x)

        x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.concatenate([avg_pool, max_pool])

        preds = tf.keras.layers.Dense(output_size, activation="sigmoid")(x)

        model = tf.keras.Model(input, preds)

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
        return model

    def loadClassifier(self):
        new_ldamodels = LdaModel.load('/Users/steviechen1982/Documents/NPS/output/model/nps_lda_model_%s' % (self.currdate))
        # new_ldamodels.update(new_corpus)
        return new_ldamodels
