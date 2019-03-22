import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
	
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def numpynotworking(sequencelist):
    X = []
    y = []
    for val in sequencelist:
        X.append(val[:-1])
        y.append(val[-1])
    return X, y


if __name__ == "__main__":
    doc = load_doc('sequencefile.txt')
    lines = doc.split('\n')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    vocab_size = len(tokenizer.word_index)+1
    sequences = np.array(sequences)
    X,y = numpynotworking(sequences)
    y=y[:9100]
    y = to_categorical(y, num_classes=vocab_size)
    X=X[:9100]
    seq_length = len(X[1])
    

	
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(250, return_sequences=True))
    model.add(LSTM(250))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
	
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(np.array(X), y, batch_size=300, epochs=5)
    
    model.save('model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
