from string import punctuation
from os import listdir
from pickle import dump
import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'rb')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def clean_doc(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

file_train = pd.read_csv('train.tsv', header=0, delimiter="\t", quoting=3)
doc = list()
for i in file_train["Sentence"]:
    doc.append(clean_doc(i))
xtrain = doc 
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
save_dataset([xtrain,ytrain], 'train.pkl')

