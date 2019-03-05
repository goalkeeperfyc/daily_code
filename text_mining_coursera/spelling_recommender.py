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


import nltk
import operator
import time
from nltk.corpus import words

nltk.download('words')

correct_spellings = words.words()

"""Question 9
For this recommender, your function should provide recommendations for the 
three default words provided above using the following distance metric:
Jaccard distance on the trigrams of the two words.
This function should return a list of length three: 
['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

entries=['cormulent', 'incendenece', 'validrate']
result = []
for entry in entries:
    spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
    distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), 
                                                   set(nltk.ngrams(spell, n=3)))) for spell in spell_list]
    result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])

result2 = []
for entry in entries:
    spell_list = [spell for spell in correct_spellings if len(spell) > 2]
    distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), 
                                                   set(nltk.ngrams(spell, n=3)))) for spell in spell_list]
    result2.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
#result2 = ['formule', 'ascendence', 'validate']

"""Question 10
For this recommender, your function should provide recommendations for the 
three default words provided above using the following distance metric:
Jaccard distance on the 4-grams of the two words.
This function should return a list of length three: 
['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

entries=['cormulent', 'incendenece', 'validrate']
result = []
for entry in entries:
    spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
    distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)), 
                                                   set(nltk.ngrams(spell, n=4)))) for spell in spell_list]
    result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])


"""Question 11
For this recommender, your function should provide recommendations for the 
three default words provided above using the following distance metric:
Edit distance on the two words with transpositions.
This function should return a list of length three: 
['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

start = time.time()
entries=['cormulent', 'incendenece', 'validrate']
result = []
for entry in entries:
    spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
    distance_list = [(spell, nltk.edit_distance(entry, spell, transpositions=True)) for spell in spell_list]
    result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
end = time.time() - start
    