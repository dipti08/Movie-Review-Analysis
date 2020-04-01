# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:13:19 2020

@author: DIPTI AGARWAL
"""

import pickle
from flask import Flask
from flask import request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def create_word_features(words):
    stop_words = set(stopwords.words('english'))
    useful_words = [word for word in words if word not in stop_words]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route('/', methods=['GET'])
def index():
    return "Hello there, the flask application is now LIVE. "
@app.route('/getrating', methods=['POST'])
def predict():
    # getting the review from the user through the node API
    review = request.form['review']
    words = word_tokenize(review)
    words = create_word_features(words)
    return (model.classify(words))
if __name__ == "__main__":
    app.run("127.0.0.1", "8080", debug=True)