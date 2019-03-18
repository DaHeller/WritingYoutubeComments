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
 


def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

if __name__ == "__main__":
    df = pd.read_csv("../data/UScomments.csv",nrows=500)
    corpus = df['comment_text'].values

    tokens = []
    sequences = []
    length = 50 + 1

    for document in corpus:
        tokenized = text_to_word_sequence(document)
        for word in tokenized:
            tokens.append(word)

    raw_text = ' '.join(tokens)
    # organize into sequences of characters
    length = 25
    sequences = list()
    for i in range(length, len(raw_text)):
        # select sequence of tokens
        seq = raw_text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))

        # save sequences to file
    out_filename = 'char_sequences.txt'
    save_doc(sequences, out_filename)