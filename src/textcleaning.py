import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

df = pd.read_csv("../data/UScomments.csv",nrows=500)
corpus = df['comment_text'].values

tokens = []
sequences = []
length = 50 + 1

for document in corpus:
    tokenized = text_to_word_sequence(document)
    for word in tokenized:
        tokens.append(word)
for num in range(length, len(tokens)):
    seq = tokens[num-length:num]
    line = ' '.join(seq)
    sequences.append(line)
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_doc(sequences, "sequencefile.txt")

   
    



# tokenizer = Tokenizer(num_words=None, 
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, 
# split=' ', char_level=False, oov_token=None, document_count=500)

# tokenizer.fit_on_texts(corpus)