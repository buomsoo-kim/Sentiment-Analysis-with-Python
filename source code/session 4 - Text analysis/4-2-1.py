import os
import nltk
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
words = []

files = os.listdir('aclImdb/train/pos')
for file in files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            if token not in stopWords:
                words.append(token)

files = os.listdir('aclImdb/train/neg')
for file in files:
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            if token not in stopWords:
                words.append(token)
# print(len(words))

words = nltk.FreqDist(words)
word_features = list(words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets = []
files = os.listdir('aclImdb/train/pos')[:1000]
for file in files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'pos'))

files = os.listdir('aclImdb/train/neg')[:1000]
for file in files:
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'neg'))

files = os.listdir('aclImdb/test/pos')[:1000]
for file in files:
    with open('aclImdb/test/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'pos'))

files = os.listdir('aclImdb/test/neg')[:1000]
for file in files:
    with open('aclImdb/test/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'neg'))

training_set = feature_sets[:2000]
test_set = feature_sets[2000:]
#print(len(training_set))
#print(len(test_set))

clf = nltk.NaiveBayesClassifier.train(training_set)
result = nltk.classify.accuracy(clf, test_set)*100
print('Accuracy of the Naive Bayes classification model: ', result)