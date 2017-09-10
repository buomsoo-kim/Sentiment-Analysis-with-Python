import nltk
from nltk.corpus import sentiwordnet as swn
import numpy as np
import os

def word_sentiment_calculator(word, tag):
    pos_score = 0
    neg_score = 0
    
    if 'NN' in tag and len(list(swn.senti_synsets(word, 'n')))>0:
        syn_set = list(swn.senti_synsets(word, 'n'))
    elif 'VB' in tag and len(list(swn.senti_synsets(word, 'v')))>0:
        syn_set = list(swn.senti_synsets(word, 'v'))
    elif 'JJ' in tag and len(list(swn.senti_synsets(word, 'a')))>0:
        syn_set = list(swn.senti_synsets(word, 'a'))
    elif 'RB' in tag and len(list(swn.senti_synsets(word, 'r')))>0:
        syn_set = list(swn.senti_synsets(word, 'r'))
    else:
        return (0,0)
    
    for syn in syn_set:
        pos_score += syn.pos_score()
        neg_score += syn.neg_score()
    return (pos_score/len(syn_set), neg_score/len(syn_set))

def sentence_sentiment_calculator(sent):
    tokens =  nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(tokens)
    
    pos_score = 0
    neg_score = 0
    for word, tag in pos_tags:
        pos_score += word_sentiment_calculator(word, tag)[0]
        neg_score += word_sentiment_calculator(word, tag)[1]
    return (pos_score, neg_score)

pos_files = os.listdir('aclImdb/train/pos')[:10]
neg_files = os.listdir('aclImdb/train/neg')[:10]

actual = [1] *10 + [0]*10 
predicted = []

for file in pos_files:
    with open('aclImdb/train/pos/{}'.format(file), 'r', encoding = 'utf-8') as f:
        scores = sentence_sentiment_calculator(f.read())
        
        if scores[0] >= scores[1]:
            predicted.append(1)
        else:
            predicted.append(0)
        f.close()

for file in neg_files:
    with open('aclImdb/train/neg/{}'.format(file), 'r', encoding = 'utf-8') as f:
        scores = sentence_sentiment_calculator(f.read())
        
        if scores[0] >= scores[1]:
            predicted.append(1)
        else:
            predicted.append(0)
        f.close()

correct = 0
incorrect = 0
for i in range(20):
    if actual[i] == predicted[i]:
        correct += 1
    else:
        incorrect += 1    

print(actual)
print(predicted)

print('Number of correct instances: ', correct)
print('Number of incorrect instances: ', incorrect)