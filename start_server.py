#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, request
from distutils.util import strtobool

from src.main.handler.Classification import ClassificationHandler
from src.main.handler.Summarize import SummarizeHandler
from src.main.utils import ConfigurationUtil
from src.main.handler.HelloWorldHandler import HelloWordHandler
from src.main.handler.TrainingModels import TrainingHandler


settings = {
    'host': str(ConfigurationUtil.get("NPS", "host")),
    'debug': strtobool(ConfigurationUtil.get("NPS", "debug_mode")),
    'port': int(ConfigurationUtil.get("NPS", "port"))
}

app = Flask(__name__)

@app.route('/nps/test', methods=['GET'])
def helloword(): return HelloWordHandler().helloWord()

@app.route('/nps/train', methods=['GET'])
def train():
    content = request.json
    return TrainingHandler().trainingProcess(content['senti_type'], content['topic_num'])

@app.route('/nps/summarize', methods=['GET'])
def predict():
    content = request.json
    return SummarizeHandler().summWeekly(content['senti_type'],content['month'])

@app.route('/nps/classifyTrain', methods=['GET'])
def classifyTrain():
    # content = request.json
    return ClassificationHandler().trainingProcess()

@app.route('/nps/predictSingle', methods=['POST'])
def classifyPredictSingle():
    content = request.json
    return ClassificationHandler().predictSingle(content['sentence'])

if __name__ == "__main__":
    app.run(host=settings['host'], port=settings['port'], debug=settings['debug'])