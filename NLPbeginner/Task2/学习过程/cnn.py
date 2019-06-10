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

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
file_train = pd.read_csv('train.tsv', header=0, delimiter="\t", quoting=3)
doc = list()
for i in file_train["Sentence"]:
    doc.append(clean_doc(i))
train_docs = doc
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
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
print(Xtrain.shape)
# load all test reviews
file_test = pd.read_csv('test.tsv', header=0, delimiter="\t", quoting=3)
text_test = ""
j = 0
doc = list()
for i in file_test["Sentence"]:
    doc.append(clean_doc(i))
test_docs = doc
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.9
 
epochs = 100
batch_size = 30


# define model
inputs = Input(shape=(max_length,), dtype='int32')
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
reshape = Reshape((max_length,embedding_dim,1))(embedding)
 
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
 
maxpool_0 = MaxPool2D(pool_size=(max_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(max_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
 
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=3, activation='softmax')(dropout)
 
# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
 
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
print("Traning Model...")

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