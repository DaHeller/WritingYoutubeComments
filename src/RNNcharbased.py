import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
 
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
def convertsequences(raw_text):
	indivchars = sorted(list(set(raw_text))) #Set to get each indiv char
	lookupdict = dict((c,i) for i,c in enumerate(indivchars)) #create dict with each char representing a int
	sequences = list()
	for line in raw_text.split('\n'):
		encoded_seq = [lookupdict[char] for char in line]
		sequences.append(encoded_seq)
	return lookupdict, np.array(sequences)
def prepare_sequences(sequences,vocab_size):
	X,y = sequences[:,:-1], sequences[:,-1] #split X into 10 y into 1
	#giving us a target to predict
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X] #one hot encode all of the X's
	X = np.array(sequences)
	y = to_categorical(y, num_classes=vocab_size) #one hot encode all targets
	return X,y

if __name__ == "__main__":

	raw_text = load_doc("char_sequences.txt")
	lookupdict, sequences = convertsequences(raw_text)
	vocab_size = len(lookupdict)
	X,y = prepare_sequences(sequences,len(lookupdict))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Actual Model preperation
	model = Sequential()
	model.add(LSTM(350, return_sequences=True, input_shape=(X.shape[1], X.shape[2]),kernel_initializer="he_normal",dropout=.2))#,kernel_initializer="he_normal"
	model.add(Dropout(0.2))
	model.add(LSTM(350, input_shape=(X.shape[1], X.shape[2]),kernel_initializer="he_normal",dropout=.2))
	model.add(Dropout(0.2))
	
	model.add(Dense(vocab_size, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=9, verbose=1, validation_split=.2,batch_size=64)
	
	model.save('charmodel3.h5')
	dump(lookupdict, open('lookupdict.pkl','wb'))