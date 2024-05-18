'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    d = dict()
    tag_d = dict()
    for s in train:
        for w, t in s:
            if w not in d:
                d[w] = dict()
            if t not in d[w]:
                d[w][t] = 0
            d[w][t] = d[w][t]+ 1
    for w in d.keys():
        counter = d[w]
        max_tag, max_count = None, 0
        for t in counter.keys():
            if counter[t] > max_count:
                max_count = counter[t]
                max_tag = t
            if t not in tag_d:
                tag_d[t] = 0
            tag_d[t] += counter[t]
        d[w] = max_tag
    max_frequency, most_frequent_tag = 0, None
    for t in tag_d:
        if tag_d[t] > max_frequency:
            max_frequency = tag_d[t]
            most_frequent_tag = t

    prediction = []
    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word not in d:
                new_sentence.append((word, most_frequent_tag))
            else:
                new_sentence.append((word, d[word]))
        prediction.append(new_sentence)
    return prediction


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k = 1e-5
    pi = 0.99
    a = dict()
    for s in train:
        for i in range(len(s)-1):
            t1 = s[i][1]
            t2 = s[i+1][1]
            if t1 not in a:
                a[t1] = dict()
            if t2 not in a[t1]:
                a[t1][t2] = 0
            a[t1][t2] += 1
    for t1 in a:
        numtags = sum(a[t1].values()) + k*len(a[t1].values()) + k
        for t2 in a[t1]:
            a[t1][t2] = (k+a[t1][t2]) / numtags
        a[t1]["OOV"] = k / numtags

    b = dict()
    for s in train:
        for w, t in s:
            if t not in b:
                b[t] = dict()
            if w not in b[t]:
                b[t][w] = 0
            b[t][w] += 1
    for t in b:
        numwords = sum(b[t].values()) + k*len(b[t].values()) + k
        for w in b[t]:
            b[t][w] = (k+b[t][w]) / numwords
        b[t]["OOV"] = k / numwords

    for t in b:
        for w in b[t]:
            if w == "OOV":
                print(f"b[{t}]['OOV'] = {b[t]['OOV']}")
    prediction = []
    for s in test:
        parent = [dict() for _ in range(len(s))]
        v = [dict() for _ in range(len(s))]
        for t in b:
            if t == 'START':
                v[0][t] = log(pi)
            else:
                v[0][t] = log((1-pi)/len(b.keys()))
        for i in range(1, len(s)):
            w = s[i]
            for t in b:
                max_prob, max_tag = -float("inf"), None
                for prev_tag in a:
                    if t not in a[prev_tag]:
                        if w not in b[t]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[t]["OOV"])
                        else:
                            x = v[i - 1][prev_tag] + log(a[prev_tag]["OOV"]) + log(b[t][w])
                    else:
                        if w not in b[t]:
                            x = v[i - 1][prev_tag] + log(a[prev_tag][t]) + log(b[t]["OOV"])
                        else:
                            x = v[i-1][prev_tag] + log(a[prev_tag][t]) + log(b[t][w])
                    if x > max_prob:
                        max_prob = x
                        max_tag = prev_tag
                v[i][t] = max_prob
                parent[i][t] = max_tag
        last_tag = "END"
        tags = [last_tag]
        for i in range(len(s)-2, -1, -1):
            last_tag = parent[i+1][last_tag]
            tags.append(last_tag)
        tags.reverse()
        prediction.append(list(zip(s, tags)))

    return prediction


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    



