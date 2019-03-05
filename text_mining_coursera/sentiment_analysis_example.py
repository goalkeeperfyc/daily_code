#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:00:03 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('/Users/fangyucheng/Documents/coursera/Text_Mining/Amazon_Unlocked_Mobile.csv')

"""
steps:
    1 data clean and divide data into train dataset and test dataset
    2 choose model to trans words to vector (count and TF-IDF) and fit the transform model
    3 trans words into vector
    4 choose logistic regerssion model 
    5 fit model 
    6 get predictions
    7 get roc_auc_score
    8 get feature names from transform model and get importance from lgmodel
    9 have a look at which words have high influence in positive or negative sentiment
"""
#data clean
df.dropna(inplace=True)
df = df[df['Rating'] != 3]
df['Positive_Rating'] = np.where(df['Rating'] > 3, 1, 0)
print("the mean of df['Positive_Rating] is %s" % df['Positive_Rating'].mean())
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positive_Rating'], 
                                                    random_state=0)
#print("X_train first entry:\n\n", X_train.iloc[0])
#print("\n\nX_train shape: ", X_train.shape)


#choose model and fit it
vect = CountVectorizer().fit(X_train)
feature_names = vect.get_feature_names()
print("the length of count-vector feature list is %s" % len(vect.get_feature_names()))

#trans words to vector
X_train_vectorized = vect.transform(X_train)

#choose logistic model and fit it
lg_model = LogisticRegression()
lg_model.fit(X_train_vectorized, y_train)

#predict the test data
predictions = lg_model.predict(vect.transform(X_test))

#get roc_auc_score
auc_score = roc_auc_score(y_test, predictions)
print("the score of count is %s" % auc_score)

#have a look at the high influence words
sorted_coef_index = lg_model.coef_[0].argsort()
feature_names_array = np.array(vect.get_feature_names())
sorted_coef = lg_model.coef_

#print the result and test the speed of list and array
start1 = time.time()
print('Smallest Coefs:\n{}\n'.format(feature_names_array[sorted_coef_index[:10]]))
end1 = time.time() - start1
start2 = time.time()
print('Largest Coefs: \n{}'.format(feature_names_array[sorted_coef_index[:-11:-1]]))
end2 = time.time() - start2
start3 = time.time()
print('Smallest Coefs:\n{}\n'.format([feature_names[index] for index in sorted_coef_index[:10]]))
end3 = time.time() - start3
start4 = time.time()
print('Largest Coefs: \n{}'.format([feature_names[index] for index in sorted_coef_index[:-11:-1]]))
end4 = time.time() - start4


#TF-IDF
vect2 = TfidfVectorizer(min_df=5).fit(X_train)
print("the length of TF-IDF feature list is %s" % len(vect2.get_feature_names()))

#trans data to vector
X_train_vectorized2 = vect2.transform(X_train)

#choose logistic regression model
lg_model2 = LogisticRegression()

#fit it
lg_model2.fit(X_train_vectorized2, y_train)

#get predictions
predictions2 = lg_model2.predict(vect2.transform(X_test))

#get roc_auc_score
auc_score2 = roc_auc_score(y_test, predictions2)
print("the score of TF-IDF is %s" % auc_score2)

#have a look at tfidf index
feature_names2 = np.array(vect2.get_feature_names())
sorted_tfidf_index = X_train_vectorized2.max(0).toarray()[0].argsort()
print("\nsmallest TF-IDF: \n", feature_names2[sorted_tfidf_index[: 10]])
print("\nlargest TF-IDF: \n", feature_names2[sorted_tfidf_index[-11: -1]])

#have a look at high influence words
sorted_coef_index2 = lg_model2.coef_[0].argsort()
sorted_coef2 = np.array(lg_model2.coef_)
print("\nsmallest coefs: \n", feature_names2[sorted_coef_index2[: 10]])
print("\nlargest coefs: \n", feature_names2[sorted_coef_index2[-11: -1]])

test_tfidf = lg_model2.predict(vect2.transform(['not an issue, phone is working',
                                                'an issue, phone is not working']))
print(test_tfidf)


#n-grams
vect3 = CountVectorizer(ngram_range=(1, 2), min_df=5).fit(X_train)
print("the length of n-grams feature list is %s" % len(vect3.get_feature_names()))

trans_X_train = vect3.transform(X_train)
lg_model3 = LogisticRegression().fit(trans_X_train, y_train)
score3= roc_auc_score(y_test, lg_model3.predict(vect3.transform(X_test)))
print("\nthe roc_auc_score of n-grams lg_model is %s" % score3)

feature_names3 = np.array(vect3.get_feature_names())
coef_index3 = lg_model3.coef_[0].argsort()

print("\nsmallest coefs: \n", feature_names3[coef_index3[: 10]])
print("\nlargest coefs: \n", feature_names3[coef_index3[-11: -1]])

