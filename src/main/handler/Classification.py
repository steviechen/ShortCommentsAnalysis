import pandas as pd
import numpy as np
import os
import datetime as dt
from src.main.utils.ModelingUtil import ModelingUtil
import tensorflow as tf

class ClassificationHandler():
    def __init__(self):
        super().__init__()
        self.modelUtil = ModelingUtil()
        self.currdate = dt.datetime.now().strftime('%Y%m%d')
        self.glove_path = "/Users/steviechen1982/Documents/NPS/input/embedding/glove.840B.300d.txt"
        # self.glove_path = "/Users/steviechen1982/Documents/NPS/input/embedding/glove.6B.50d.txt"
        self.checkpoint_path = "/Users/steviechen1982/Documents/NPS/output/ckp/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.classifier_path = "/Users/steviechen1982/Documents/NPS/output/model/multi_labeling_classfier.h5"

    def getEntireDF(self):
        rawdata_cla = pd.read_csv('/Users/steviechen1982/Documents/NPS/input/rawdata/NPS_DETRACTORS_rawdata.csv',
                                  encoding='latin-1')

        return rawdata_cla

    def trainingProcess(self):
        embed_size = 300
        batch_size = 128

        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            cp_callback
        ]

        model = self.modelUtil.trainMultiLabel(self.getEntireDF(), self.glove_path, embed_size, callbacks, batch_size)
        model.save(self.classifier_path)
        return 'Training process on Multi-label Multi-class is finished.'

    def predictSingle(self, sentence):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        model = tf.keras.models.load_model(self.classifier_path)

        model.load_weights(latest)

        return self.modelUtil.predictMultiLabelSingle(sentence, model)

    def predictBatch(self):
        return