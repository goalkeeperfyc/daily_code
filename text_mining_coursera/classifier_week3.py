#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:05:12 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import re
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
q1 = count_by_group['text'][1] / (count_by_group['text'][0] 
                                  + count_by_group['text'][1])

q1 = q1 * 100


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
feature_names = tfidf.get_feature_names()
#tfidf_score = tfidf.idf_
tfidf_score = trans_X_train.max(0).toarray()[0].tolist()
word_with_score_list = list(zip(feature_names, tfidf_score))
sorted_list = sorted(word_with_score_list, key=lambda x: (x[1], x[0]))
small_list = sorted_list[:20]
small_series = pd.Series([data_tup[1] for data_tup in small_list], 
                         index=[data_tup[0] for data_tup in small_list])
sorted_list2 = sorted(word_with_score_list, key=lambda x: (-x[1], x[0]), reverse=True)
large_list = sorted_list2[-20:]
large_list = list(reversed(large_list))
large_series = pd.Series([data_tup[1] for data_tup in large_list], 
                         index=[data_tup[0] for data_tup in large_list])


"""Question 5
Fit and transform the training data X_train using a Tfidf Vectorizer 
ignoring terms that have a document frequency strictly lower than 3.

Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 
and compute the area under the curve (AUC) score using the transformed test data.

This function should return the AUC score as a float.
"""

tfidf = TfidfVectorizer(min_df=3).fit(X_train)
trans_X_train = tfidf.transform(X_train)
nb_model = MultinomialNB(alpha=0.1).fit(trans_X_train, y_train)
q5 = roc_auc_score(y_test, nb_model.predict(tfidf.transform(X_test)))


"""Question 6
What is the average length of documents (number of characters) for not spam and spam documents?
This function should return a tuple (average length not spam, average length spam).
"""

spam_data['length'] = spam_data['text'].apply(lambda x: len(x))
average_not_spam = np.mean(spam_data['length'][spam_data['target'] == 0])
average_spam = np.mean(spam_data['length'][spam_data['target'] == 1])


"""Question 7
Fit and transform the training data X_train using a Tfidf Vectorizer ignoring 
terms that have a document frequency strictly lower than 5.

Using this document-term matrix and an additional feature, the length of document 
(number of characters), fit a Support Vector Classification model with 
regularization C=10000. Then compute the area under 
the curve (AUC) score using the transformed test data.

This function should return the AUC score as a float.
"""

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

tfidf = TfidfVectorizer(min_df=5).fit(X_train)
trans_X_train = tfidf.transform(X_train)
add_feature_X_train = add_feature(trans_X_train, X_train.str.len())

trans_X_test = tfidf.transform(X_test)
add_feature_X_test = add_feature(trans_X_test, X_test.str.len())

svm_model = SVC(C=10000).fit(add_feature_X_train, y_train)
q7 = roc_auc_score(y_test, svm_model.predict(add_feature_X_test))


"""Question 8
What is the average number of digits per document for not spam and spam documents?

This function should return a tuple (average # digits not spam, average # digits spam).
"""

spam_data['number_of_digits'] = spam_data['text'].apply(lambda x: 
                                                        len([dig for dig in x if dig.isdigit()]))
average_spam_digits = np.mean(spam_data['number_of_digits'][spam_data['target'] == 1])
average_not_spam_digits = np.mean(spam_data['number_of_digits'][spam_data['target'] == 0])


"""Question 9
Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms 
that have a document frequency strictly lower than 5 and using word n-grams 
from n=1 to n=3 (unigrams, bigrams, and trigrams).

Using this document-term matrix and the following additional features:

the length of document (number of characters)
number of digits per document
fit a Logistic Regression model with regularization C=100. 
Then compute the area under the curve (AUC) score using the transformed test data.

This function should return the AUC score as a float.
"""

tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train)
trans_X_train = tfidf.transform(X_train)
add_feature_X_train = add_feature(trans_X_train, [X_train.str.len(), 
                                                  X_train.apply(lambda x: len([dig for dig in x if dig.isdigit()]))])

trans_X_test = tfidf.transform(X_test)
add_feature_X_test = add_feature(trans_X_test, [X_test.str.len(), 
                                                X_test.apply(lambda x: len([dig for dig in x if dig.isdigit()]))])
    
lg_model = LogisticRegression(C=100).fit(add_feature_X_train, y_train)
score9 = roc_auc_score(y_test, lg_model.predict(add_feature_X_test)) 

"""Question 10
What is the average number of non-word characters (anything other than a letter, 
digit or underscore) per document for not spam and spam documents?

Hint: Use \w and \W character classes

This function should return a tuple (average # non-word characters not spam, 
average # non-word characters spam).
"""

spam_data['non_word1'] = spam_data['text'].str.findall('\W').str.len()
average_spam = np.mean(spam_data['non_word1'][spam_data['target'] == 1])
average_not_spam = np.mean(spam_data['non_word1'][spam_data['target'] == 0])

#spam_data['non_word2'] = spam_data['text'].str.findall('\w').str.len()
#spam_data['non_word3'] = spam_data['text'].str.findall(r'(\W)').str.len()

"""Question 11
Fit and transform the training data X_train using a Count Vectorizer ignoring terms 
that have a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.

To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' 
which creates character n-grams only from text inside word boundaries. 
This should make the model more robust to spelling mistakes.

Using this document-term matrix and the following additional features:

the length of document (number of characters)
number of digits per document
number of non-word characters (anything other than a letter, digit or underscore.)

fit a Logistic Regression model with regularization C=100. 
Then compute the area under the curve (AUC) score using the transformed test data.

Also find the 10 smallest and 10 largest coefficients from the model 
and return them along with the AUC score in a tuple.

The list of 10 smallest coefficients should be sorted smallest first, 
the list of 10 largest coefficients should be sorted largest first.

The three features that were added to the document term matrix should have 
the following names should they appear in the list of coefficients: 
    ['length_of_doc', 'digit_count', 'non_word_char_count']

This function should return a tuple (AUC score as a float, 
smallest coefs list, largest coefs list).
"""

cove = CountVectorizer(min_df=5, ngram_range=(2, 5), 
                       analyzer='char_wb').fit(X_train)
trans_X_train = cove.transform(X_train)
add_feature_X_train = add_feature(trans_X_train, [X_train.str.len(),
                                                  X_train.apply(lambda x: len([dig for dig in x if dig.isdigit()])),
                                                  X_train.str.findall('\W').str.len()])

trans_X_test = cove.transform(X_test)
add_feature_X_test = add_feature(trans_X_test, [X_test.str.len(),
                                                X_test.apply(lambda x: len([dig for dig in x if dig.isdigit()])),
                                                X_test.str.findall('\W').str.len()])
lg_model = LogisticRegression(C=100).fit(add_feature_X_train, y_train)
score = roc_auc_score(y_test, lg_model.predict(add_feature_X_test))

feature_name_list = cove.get_feature_names()
add_feature_list = ['length_of_doc', 'digit_count', 'non_word_char_count']
feature_name_list.extend(add_feature_list)
coef = lg_model.coef_[0].tolist()
feature_coef_list = list(zip(feature_name_list, coef))

sorted_list = sorted(feature_coef_list, key=operator.itemgetter(1))

small_list = sorted_list[:10]
large_list = sorted_list[-10:]
large_list = list(reversed(large_list))

small_list_result = [line[0] for line in small_list]
large_list_result = [line[0] for line in large_list]