from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import BatchNormalization
import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding
 
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
file_train = pd.read_csv('train.tsv', header=0, delimiter="\t", quoting=3)
doc = list()
for i in file_train["Sentence"]:
    doc.append(clean_doc(i,vocab))
train_docs = doc
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

# pad sequences
max_length = max([len(s.split()) for s in file_train["Sentence"]])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(Xtrain.shape)

# define training labels
y = file_train["Sentiment"]
ytrain = []
NumberofSize = file_train["Sentiment"].size
for i in list(range(0,NumberofSize)):
        if file_train["Sentiment"][i] == 'negative' :
            ytrain.append(0)
        if file_train["Sentiment"][i] == 'neutral' :
            ytrain.append(1)
        if file_train["Sentiment"][i] == 'positive' :
            ytrain.append(2)
ytrain = np.array(ytrain)

from keras.utils.np_utils import to_categorical
ytrain = to_categorical(ytrain,3)

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# load all test reviews

file_test = pd.read_csv('test.tsv', header=0, delimiter="\t", quoting=3)
doc = list()
for i in file_test["Sentence"]:
    doc.append(clean_doc(i,vocab))
test_docs = doc
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, validation_split = 0.2 ,epochs=50,verbose=2)
# evaluate
loss, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Test Accuracy: %f' % (acc*100))

result = model.predict(Xtest)
temp = np.argmax(result, axis=1)
tran_result = []
print(result)
#print(shape(result))
for i in temp:
    #print(result[i])
    if i == 0 :
        tran_result.append('negative')
        continue
    if i == 1 :
        tran_result.append('neutral')
        continue
    if i == 2:
        tran_result.append('positive')
        continue
    #print(result[i])
output = pd.DataFrame( data={"id":file_test["ID2"],"polarity":tran_result} )
output.to_csv( "final.csv", index=False, quoting=3 )