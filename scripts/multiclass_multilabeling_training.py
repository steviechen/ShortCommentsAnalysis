import numpy as np
import pandas as pd

from gensim.utils import simple_preprocess
import tensorflow as tf
import pickle as pkl
import os

proj_dir = '/home/wbxbuilds/nps_analysis/data/'

rawdata_cla = pd.read_csv(proj_dir + 'input/rawdata/NPS_DETRACTORS_rawdata.csv', encoding='latin-1')
data_cla = pd.concat([rawdata_cla[["siteid", "cliversion", "domainname", "nps_score", "concat_comments", "Text", "words list"]], \
           pd.get_dummies(rawdata_cla["join_issue"]), pd.get_dummies(rawdata_cla["audio_issue"]), \
           pd.get_dummies(rawdata_cla["video_issue"]), pd.get_dummies(rawdata_cla["sharing_issue"]), \
           pd.get_dummies(rawdata_cla["other_issue"])], axis=1)


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


# TRAIN_DATA = "data/train.csv"
GLOVE_EMBEDDING = proj_dir + "input/embedding/glove.840B.300d.txt"

# train = pd.read_csv(TRAIN_DATA)

# train["comment_text"].fillna("fillna")

x_train = rawdata_cla["concat_comments"].str.lower()
y_train = data_cla[data_cla.columns[7:]].values

max_words = 5000
max_len = 200

embed_size = 300

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)

tokenizer.fit_on_texts(x_train)

with open(proj_dir + '/output/model/tokenizer.pkl', 'wb') as pickle_file:
    pkl.dump(tokenizer, pickle_file)

x_train = tokenizer.texts_to_sequences(x_train)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)

embeddings_index = {}

with open(GLOVE_EMBEDDING, encoding='utf8') as f:
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

input = tf.keras.layers.Input(shape=(max_len,))

x = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x)

x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.concatenate([avg_pool, max_pool])

preds = tf.keras.layers.Dense(18, activation="sigmoid")(x)

model = tf.keras.Model(input, preds)

model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

batch_size = 64

checkpoint_path = proj_dir + "output/ckp/cp_1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    cp_callback
]

model.fit(x_train, y_train, validation_split=0.3, batch_size=batch_size, epochs=50, callbacks=callbacks, verbose=1)

model.save(proj_dir + 'output/model/multi_labeling_classfier.h5')
