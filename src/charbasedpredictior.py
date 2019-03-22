from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np 

def generate_seq(model,mapping,seq_length,seed_text,n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded],maxlen=seq_length, truncating="pre")[0]
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = np.array(encoded)
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        yhat = model.predict_classes(encoded)
        out_char = ''
        for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
        in_text += char
    return in_text
# load the model
model = load_model('charmodel3.h5')
# load the mapping
mapping = load(open('lookupdict.pkl', 'rb'))
 
print(generate_seq(model, mapping, 15, 'global war', 100))
# print("2")
# print(generate_seq(model, mapping, 15, 'icecream 1', 100))
# print("2")
# print(generate_seq(model, mapping, 15, 'time squar', 100))
# print("2")
# print(generate_seq(model, mapping, 15, 'dreams log', 100))
# print("2")
# print(generate_seq(model, mapping, 15, 'iphone 100', 100))
