#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # try en_roberta_large_nli_stsb_mean_tokens
import csv
import sys
from typing import List

# import gensim
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
# splitter = Splitter()

import re
from nltk.corpus import stopwords
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from nltk.stem import WordNetLemmatizer

import spacy
from sense2vec import Sense2VecComponent


from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_lg")
s2v = nlp.add_pipe("sense2vec")
lemmatizer = nlp.get_pipe("lemmatizer")

import pandas as pd
from lxml import html
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.ensemble import RandomForestClassifier
import gensim
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter,defaultdict
from string import punctuation
# from razdel import tokenize as razdel_tokenize
import os
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
# %matplotlib inline

morph = MorphAnalyzer()
punct = punctuation+'«»—…“”*№–'

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords


class Movie:
    def __init__(self, id, name, description): # str
        self.id = id
        self.name = name
        self.description = description
        self.data = self.process_description(self.description) #data - processed description

    def get_score(self, desc): # str в float
        query_embedding = model.encode(desc)
        passage_embedding = model.encode(self.data)
        return util.pytorch_cos_sim(query_embedding, passage_embedding)

    def process_description(self, desc):
        return desc.lower()

    def __str__(self):   # this function is called whenever you call print on an instance of this class
        return "id: {}, name: {}, desc: {}".format(self.id, self.name, self.data) 
    
    def __repr__(self): # this function is called whenever you call print on a list of instances of this class (print movies)
        return self.__str__()


def main(filename): #str
    movies: List[Movie] = [] # create a list of obejcts (Movie type)

    with open(filename, 'r', encoding='utf-8', newline='') as file: #open file excluding errors chapman
        reader = csv.DictReader(file, delimiter=',')
        for line in reader: #for every line -> every film
            movie = Movie(line['id'], line['title'], line['overview']) #create movie and add to the list.
            movies.append(movie)

    while True: 
        x = input("Please enter the description: ") #x goes to get_score, and x becomes desc
        if x == 'end':
            break

        res = [] 
        for movie in movies: 
            res.append({'movie': movie.name, 'score': movie.get_score(x)}) 

        res.sort(reverse=True, key=lambda e: e['score'])

        print(res[:10])


main("IMDB-Movie-Data.csv")

