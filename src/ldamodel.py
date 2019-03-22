import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.feature_extraction import text
from wordcloud import WordCloud



def createvocab(corpus):
    lookup={}
    stopwords = text.ENGLISH_STOP_WORDS.union({'com','www', 'https','ve', 'like','ni', 'video', 'just'})
    vectorizer = CountVectorizer(max_features=1000, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def display_topics(phi, feature_names, num_top_words):
    topics = dict()
    for topic_idx, topic in enumerate(phi):
        topics[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])
    return topics

def elbowplotlda(listoftopics, vectorizedcorpus):
    perplexitylst = []
    log_likelihood = []
    for num in listoftopics:
        model = LatentDirichletAllocation(n_jobs=-1, n_components=num)
        model.fit(vectorizedcorpus)
        log_likelihood.append(model.score(vectorizedcorpus))
        perplexitylst.append(model.perplexity(vectorizedcorpus))
    print(log_likelihood)
    print(perplexitylst)
    plt.plot(listoftopics, log_likelihood, '-', label="log_likelihood")
    #plt.plot(listoftopics, perplexitylst, '-', label='perplexitylst')
    plt.legend()
    plt.savefig("../images/elbowplot")
    plt.show()

def makewordclouds(topic_names):
    bark = len(topic_names)
    for num in range(0, bark):
        Text = topic_names[num]
        print(Text)
        wordcloud = WordCloud().generate(Text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        temptitle='Topic {} Word Cloud'.format(num)
        plt.title(temptitle)
        tempsaveloc = '../images/topic{}_wrdcld.png'.format(num)
        plt.savefig(tempsaveloc)
        plt.show()








if __name__ == "__main__":
    df = pd.read_csv("../data/UScomments.csv",nrows=50000)
    corpus = df['comment_text'].values
    X, vectorizer = createvocab(corpus)
    ldamodel = LatentDirichletAllocation(n_jobs=-1, n_components=5)
    ldamodel.fit(X)
    feature_names = vectorizer.get_feature_names()
    phi = ldamodel.components_
    topic_names = display_topics(phi, feature_names, 10)
    topic_names2 = list(topic_names.values())
    makewordclouds(topic_names2)
    