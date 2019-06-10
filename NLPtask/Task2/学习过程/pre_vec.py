from string import punctuation
from os import listdir
from gensim.models import Word2Vec
import pandas as pd
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
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data

file_train = pd.read_csv('train.tsv', header=0, delimiter="\t", quoting=3)
doc = list()
for i in file_train["Sentence"]:
    print(doc_to_clean_lines(i,vocab))
    doc += (doc_to_clean_lines(i,vocab))

file_test = pd.read_csv('test.tsv', header=0, delimiter="\t", quoting=3)
for i in file_test["Sentence"]:
    doc += (doc_to_clean_lines(i,vocab))
sentences = doc
print('Total training sentences: %d' % len(sentences))

# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)