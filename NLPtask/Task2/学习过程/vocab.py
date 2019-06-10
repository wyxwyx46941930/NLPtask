from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(doc,vocab):
    	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

file_train = pd.read_csv('train.tsv', header=0, delimiter="\t", quoting=3)
text_train = ""
for i in file_train["Sentence"]:
    text_train += i
    text_train += '\n'

file_test = pd.read_csv('test.tsv', header=0, delimiter="\t", quoting=3)
text_test = ""
for i in file_test["Sentence"]:
    text_test += i
    text_test += '\n'

# define vocab
vocab = Counter()
# add all docs to vocab
add_doc_to_vocab(text_train,vocab)
add_doc_to_vocab(text_test,vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))

# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')