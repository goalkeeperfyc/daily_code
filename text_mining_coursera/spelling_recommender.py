#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:09:11 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

"""
For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in correct_spellings that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

*Each of the three different recommenders will use a different distance measure (outlined below).

Each of the recommenders should provide recommendations for the three default words provided: ['cormulent', 'incendenece', 'validrate'].
"""

import os
import nltk
import operator
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import words

correct_spellings = words.words()

"""Question 9
For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

Jaccard distance on the trigrams of the two words.

This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    
    return # Your answer here
    
answer_nine()


"""Question 10
For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

Jaccard distance on the 4-grams of the two words.

This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

"""Question 11
For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

Edit distance on the two words with transpositions.

This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""