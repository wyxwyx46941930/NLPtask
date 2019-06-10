import pandas as pd
train = pd.read_csv("train.tsv", header=0, delimiter="\t", quoting=3)
train.columns.values
train = train.sample(frac=1)
train= train.reset_index(drop=True)
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))
new_train = []
NumberofSize = train["Phrase"].size
for i in list(range(0,NumberofSize)):
    if i % 1000 == 0:
        print('{}/{}'.format(i,NumberofSize))
    if i < 10000:
        new_train.append(review_to_words(train["Phrase"][i]))
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1))
vect.fit(new_train)
X_data = vect.fit_transform(new_train).toarray()
####
import numpy as np
class SoftMax:
    def __init__(self, maxstep=1000, C=1e-4, alpha=0.4):
        self.maxstep = maxstep
        self.C = C 
        self.alpha = alpha  
        self.w = None  
        self.L = None  
        self.D = None 
        self.N = None 
    def init_param(self, X_data, y_data):
        b = np.ones((X_data.shape[0], 1))
        X_data = np.hstack((X_data, b))  
        self.L = len(np.unique(y_data))
        self.D = X_data.shape[1]
        self.N = X_data.shape[0]
        self.w = np.ones((self.L, self.D))  
        return X_data
    def bgd(self, X_data, y_data):
        step = 0
        while step < self.maxstep:
            step += 1
            if step % 100 == 0:
                print('now step is :',step);
            prob = np.exp(X_data @ self.w.T) 
            nf = np.transpose([prob.sum(axis=1)])
            nf = np.repeat(nf, self.L, axis=1)  
            prob = -prob / nf  
            for i in range(self.N):
                prob[i, int(y_data[i])] += 1
            grad = -1.0 / self.N * prob.T @ X_data + self.C * self.w  
            self.w -= self.alpha * grad
        return
    def fit(self, X_data, y_data):
        X_data = self.init_param(X_data, y_data)
        self.bgd(X_data, y_data)
        return
    def predict(self, X):
        b = np.ones((X.shape[0], 1))
        X = np.hstack((X, b))  
        prob = np.exp(X @ self.w.T)
        return np.argmax(prob, axis=1)
if __name__ == '__main__':
    y_data = []
    NumberofSize = train["Sentiment"].size
    for i in list(range(0,NumberofSize)):
        if i < 10000: 
            y_data.append(train["Sentiment"][i])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
    # softmax 训练与测试
    #clf = SoftMax(maxstep=1000, alpha=0.05, C=1e-4)
    #clf.fit(X_train, y_train)
    # 随机森林 训练与测试
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit( X_train, y_train )
    y_pred = forest.predict(X_test)
    score = 0
    for y, y_pred in zip(y_test, y_pred):
        score += 1 if y == y_pred else 0
    print('The Accuracy of correction set is {}'.format(score / len(y_test)))

    # 完成测试集合的预测
    test = pd.read_csv("test.tsv", header=0, delimiter="\t", quoting=3 )
    
    num_reviews = len(test["Phrase"])
    clean_test_reviews = [] 

    for i in list(range(0,num_reviews)):
        if( (i+1) % 1000 == 0 ):
            print ( "Review %d of %d\n" % (i+1, num_reviews))
        clean_review = review_to_words( test["Phrase"][i] )
        clean_test_reviews.append( clean_review )
    test_data_features = vect.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame( data={"PhraseId":test["PhraseId"], "Sentiment":result} )
    output.to_csv( "final.csv", index=False, quoting=3 )
    

