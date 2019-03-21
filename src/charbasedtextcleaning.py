from numpy import array
import numpy as np
from pickle import dump
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
 
def create_doc(corpus,length_of_seqs):
    #tokenslist = corpustolist(corpus)
    tokenslist = cleancorpustolist(corpus)
    sequences = createsequences(tokenslist,length_of_seqs)
    return sequences
    
def corpustolist(corpus):
    tokens = []
    for document in corpus:
        templist = text_to_word_sequence(document) #Keras built inbreaks strings to into indivdual words and cleans some text
        for word in templist:
            tokens.append(word) #appending to tokens list to create one giant list of tokens
    return tokens
def createsequences(tokenslist, length_of_seqs):
    raw_text = ' '.join(tokenslist)
    sequences = []
    for i in range(length_of_seqs, len(raw_text)):
        seq = raw_text[i-length_of_seqs:i+1] #select 10 characters at a time
        sequences.append(seq)
    print('Total Sequences: {}'.format(len(sequences)))
    return sequences
def cleancorpustolist(corpus):
    s = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ', ''}
    tokens = []
    for document in corpus:
        templist = text_to_word_sequence(document)
        for word in templist:
            tempwordlist=list(word)
            boolval = False
            for letter in tempwordlist:
                if letter not in list(s):
                    boolval=True
            if boolval == False:
                tokens.append(word)
    return tokens

def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()



if __name__ == "__main__":
    df = pd.read_csv("../data/UScomments.csv",nrows=50000)
    df = df[df['video_id']=="WYYvHb03Eog"] #pulling comments from just 1 unique video id
    corpus = df['comment_text'].values

    sequences = create_doc(corpus, 15)
    #save_doc(sequences, 'char_sequences.txt')