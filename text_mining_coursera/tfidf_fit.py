#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:54:28 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
X2 = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)
print(X2.shape)