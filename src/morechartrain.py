from keras.models import load_model
import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from RNNcharbased import load_doc, convertsequences, prepare_sequences




if __name__ == "__main__":
    model = load_model('charmodel.h5')
    raw_text = load_doc("char_sequences.txt")
    lookupdict, sequences = convertsequences(raw_text)
    vocab_size = len(lookupdict)

    X,y = prepare_sequences(sequences,len(lookupdict))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train, epochs=5, verbose=1, validation_split=.2)
