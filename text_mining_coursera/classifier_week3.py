#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:05:12 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

spam_data = pd.read_csv('/Users/fangyucheng/Documents/coursera/Text_Mining/spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

"""Question 1
What percentage of the documents in spam_data are spam?
This function should return a float, the percent value (ratio*100).
"""

count_by_group = spam_data.groupby('target').count()
q1 = count_by_group['text'][0] / (count_by_group['text'][0] 
                                  + count_by_group['text'][1])


"""Question 2
Fit the training data X_train using a Count Vectorizer with default parameters.
What is the longest token in the vocabulary?
This function should return a string.
"""

vect = CountVectorizer().fit(X_train)
feature_names = vect.get_feature_names()
sorted_list = sorted(feature_names, key=len, reverse=True)
q2 = sorted_list[0]


"""Question 3
Fit and transform the training data X_train using a Count Vectorizer with default parameters.
Next, fit a fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1. 
Find the area under the curve (AUC) score using the transformed test data.
This function should return the AUC score as a float.
"""

trans_X_train = vect.transform(X_train)
multi_model = MultinomialNB(alpha=0.1).fit(trans_X_train, y_train)
trans_X_test = vect.transform(X_test)
predictions = multi_model.predict(trans_X_test)
q3 = roc_auc_score(y_test, predictions)


"""Question 4
Fit and transform the training data X_train using a Tfidf Vectorizer with default parameters.
What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
Put these features in a two series where each series is sorted by tf-idf value 
and then alphabetically by feature name. The index of the series should be the feature name, 
and the data should be the tf-idf.
The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, 
the list of 20 features with largest tf-idfs should be sorted largest first.
This function should return a tuple of two series (smallest tf-idfs series, largest tf-idfs series).
"""

tfidf = TfidfVectorizer().fit(X_train)
trans_X_train = tfidf.transform(X_train)

tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(X_train)

"""Question 5
Fit and transform the training data X_train using a Tfidf Vectorizer 
ignoring terms that have a document frequency strictly lower than 3.
Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 
and compute the area under the curve (AUC) score using the transformed test data.
This function should return the AUC score as a float.
"""