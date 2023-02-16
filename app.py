# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:26:10 2023

@author: Niki
"""

import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
import os
from  tensorflow.keras.preprocessing.text import *

nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")


path = os.path.dirname(__file__)





tokenizer = pickle.load(open(path + '/tokenizer.pickle', 'rb'))

lem = pickle.load(open(path + '/lemmatizer.pickle', 'rb'))

model = tf.keras.models.load_model(path + "/spotify_sentiment_analysis")




def predict_sentiment(review):
    
    review = re.sub("[^a-zA-Z0-9-+ ]"," ",review)
    review = review.lower()
    review = review.split()
    review = " ".join(x for x in review if x not in set(stopwords.words("english")))
    review = " ".join(lem.lemmatize(word) for word in review.split())
    
    review = tokenizer.texts_to_sequences([review])
    review = tf.keras.preprocessing.sequence.pad_sequences(review, maxlen = 30, truncating = "post", padding = "post")
    
    pred = model.predict(review)
    
    if (np.argmax(pred) == 0):
        return("negative")
    elif (np.argmax(pred) == 1):
        return("neutral")
    elif (np.argmax(pred) == 2):
        return("positive")
    
     
    

def main():
    #st.title("Spotify sentiment analysis")
    st.subheader("Find how the Review is :")
    html_temp = """
    <div>
    <h2> Spotify sentiment analysis </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    review = st.text_input("review", "Type here")
    result = ""
    if st.button("predict") :
        result = predict_sentiment(review)
    st.success("The review is {}".format(result))
    

    
if __name__ == "__main__":
    main()