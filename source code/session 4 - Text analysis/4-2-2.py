import os
import nltk
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

feature_sets = []
files = os.listdir('aclImdb/train/pos')
for file in files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'pos'))

files = os.listdir('aclImdb/train/neg')
for file in files:
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'neg'))

files = os.listdir('aclImdb/test/pos')
for file in files:
    with open('aclImdb/test/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'pos'))
        
files = os.listdir('aclImdb/test/neg')
for file in files:
    with open('aclImdb/test/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'neg'))

training_set = feature_sets[:25000]
test_set = feature_sets[25000:]        

clf = nltk.NaiveBayesClassifier.train(training_set)
result = nltk.classify.accuracy(clf, test_set)*100

print('Accuracy of the Naive Bayes classification model: ', result)