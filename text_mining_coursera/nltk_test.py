#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:16:53 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import os
import nltk
import operator
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

path = '/Users/fangyucheng/Documents/code/python_code/daily_code/text_mining_coursera/'
os.chdir(path)


# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()

# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

def example_one():
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

def example_two():
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))


def example_three():
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]
    return len(set(lemmatized))


"""
Question 1
What is the lexical diversity of the given text input? 
(i.e. ratio of unique tokens to the total number of tokens)
"""

def answer_one():
    return example_two() / example_one()

answer_one()


"""
Question 2
What percentage of tokens is 'whale'or 'Whale'?
"""

answer = defaultdict(int)
for line in text1:
    answer[line] += 1

whale = answer['whale']
Whale = answer['Whale']
total = whale + Whale
result2 = total / len(nltk.word_tokenize(moby_raw)) * 100

"""
Question 3
What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
This function should return a list of 20 tuples where each tuple is of the form (token, frequency). 
The list should be sorted in descending order of frequency.
"""

final_list = []
answer_dict = dict(answer)
for key, value in answer_dict.items():
    new_dict = {"word": key,
                "count": value}
    final_list.append(new_dict)
    
df = pd.DataFrame(final_list)
df = df.sort_values(['count'], ascending=False)
result = df[:20]
word_list = result['word'].tolist()
count_list = result['count'].tolist()
result_list = []
count_max = len(word_list)

for count in range(0, count_max):
    result_tup = (word_list[count], count_list[count])
    result_list.append(result_tup)
    
"""
Question 4
What tokens have a length of greater than 5 and frequency of more than 150?
This function should return an alphabetically sorted list of the tokens that match the above constraints. 
To sort your list, use sorted()
"""

result_4_list = []

for word_dict in final_list:
    if len(word_dict['word']) > 5 and word_dict['count'] > 150:
        result_4_list.append(word_dict['word'])

"""
Question 5
Find the longest word in text1 and that word's length.
This function should return a tuple (longest_word, length).
"""
result_5_tup = ()

for line in final_list:
    line['length'] = len(line['word'])

df = pd.DataFrame(final_list)
df = df.sort_values(['length'], ascending=False)
result_5 = df[:1]

length = result_5['length'].tolist()
word = result_5['word'].tolist()

result_5_tup = (word[0], length[0])

"""
Question 6
What unique words have a frequency of more than 2000? What is their frequency?
"Hint: you may want to use isalpha() to check if the token is a word and not punctuation."
This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.
"""

result_6_list = []
selected_list = []

for line in final_list:
    if line['word'].isalpha() is True and line['count'] > 2000:
        selected_list.append(line)

df = pd.DataFrame(selected_list)
df = df.sort_values(['count'], ascending=False)

length_list = df['count'].tolist()
word_list = df['word'].tolist()

count_max = len(word_list)
for count in range(0, count_max):
    result_tup = (word_list[count], count_list[count])
    result_6_list.append(result_tup)

    
"""
Question 7
What is the average number of tokens per sentence?
This function should return a float.
"""        

sentence_number = len(nltk.sent_tokenize(moby_raw))
word_number = len(nltk.word_tokenize(moby_raw))
average = word_number / sentence_number


"""
Question 8
What are the 5 most frequent parts of speech in this text? What is their frequency?
This function should return a list of tuples of the form (part_of_speech, frequency) 
sorted in descending order of frequency.
"""

get_tag = nltk.pos_tag(text1)
tag_list = [line[1] for line in get_tag ]
count_dict = Counter(tag_list)
result = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)[:5] 
