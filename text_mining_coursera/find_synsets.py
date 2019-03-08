#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:30:10 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import nltk
from nltk.corpus import wordnet as wn

def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

doc = 'Fish are nvqjp friends.'

def doc_to_synsets(doc):
    
    #分词
    tokens = nltk.word_tokenize(doc)
    #标注词性
    pos = nltk.pos_tag(tokens)
    #转换词性的名字，从pos_tag的名字到wordnet的名字
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]
    ans = list(zip(tokens, wntag))
    sets = [wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
    return final

s1 = doc_to_synsets('I like cats')
s2 = doc_to_synsets('I like dogs')
s = []
for i1 in s1:
    r = []
    scores = [x for x in [i1.path_similarity(i2) for i2 in s2] if x is not None]
    if scores:
        s.append(max(scores))

result = sum(s)/len(s)