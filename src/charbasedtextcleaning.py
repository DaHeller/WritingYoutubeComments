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
    tokenslist = corpustolist(corpus)
    sequences = createsequences(tokenslist,length_of_seqs)
    return sequences
    
def corpustolist(corpus):
    tokens = []
    for document in corpus:
        templist = text_to_word_sequence(document) #Keras built inbreaks strings to into indivdual words and clears some punctiation
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
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()



if __name__ == "__main__":
    df = pd.read_csv("../data/UScomments.csv",nrows=50000)
    df = df = df.sample(n=500, random_state=420)
    corpus = df['comment_text'].values

    sequences = create_doc(corpus, 15)
    save_doc(sequences, 'char_sequences.txt')