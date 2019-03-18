import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.feature_extraction import text



def createvocab(corpus):
    lookup={}
    stopwords = text.ENGLISH_STOP_WORDS.union({'com','www', 'https','ve', 'like','ni'})
    vectorizer = CountVectorizer(max_df=.85, min_df=2, max_features=1000, stopwords=stopwords)
    X = vectorizer.fit_transform(corpus)
    for idx,word in enumerate(X):
        lookup[idx] = word


def display_topics(phi, feature_names, num_top_words):
    topics = dict()
    for topic_idx, topic in enumerate(phi):
        topics[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])
    return topics







if __name__ == "__main__":
    df = pd.read_csv("../data/UScomments.csv",nrows=50000)
    corpus = df['comment_text'].values
    stopwords = text.ENGLISH_STOP_WORDS.union({'com','www', 'https','ve', 'like','ni'})
    vectorizer = CountVectorizer(max_df=.85, min_df=2, 
    max_features=1000, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    
    ldamodel = LatentDirichletAllocation(n_jobs=-1, n_components=8)
    ldamodel.fit(X)
    feature_names = vectorizer.get_feature_names()
    phi = ldamodel.components_
    display_topics(phi, feature_names, 10)