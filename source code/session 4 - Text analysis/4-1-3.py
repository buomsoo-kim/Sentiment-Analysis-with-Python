import nltk
from nltk.corpus import sentiwordnet as swn
import numpy as np

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

with open('result-1-3-4.txt', 'r', encoding = 'utf-8') as f:
    all_reviews = f.readlines()
    f.close()

scores = []
for review in all_reviews:
    scores.append(sentence_sentiment_calculator(review))
scores = np.array(scores)

mean_scores = np.mean(scores, axis = 0)
print(mean_scores)