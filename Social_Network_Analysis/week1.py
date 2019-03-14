#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 01:24:54 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);

G_df = pd.read_csv('Employee_Movie_Choices.txt')
G_df['employee'] = G_df['#Employee\tMovie'].apply(lambda x: x.split('\t')[0])
G_df['movie'] = G_df['#Employee\tMovie'].apply(lambda x: x.split('\t')[1])
del G_df['#Employee\tMovie']

def answer_one():
    G = nx.from_pandas_edgelist(G_df, 'employee', 'movie')
    return G

answer1 = answer_one()

employee_list = G_df['employee'].tolist()
movie_list = G_df['movie'].tolist()

def answer_two():
    G = answer_one()
    for employee in employee_list:
        G.node[employee]['type'] = 'employee'
    for movie in movie_list:
        G.node[movie]['type'] = 'movie'
    return G

answer2 = answer_two()


def answer_three():
    G = answer_two()
    X = set(employee_list)
    P = bipartite.weighted_projected_graph(G, X)    
    return P

answer3 = answer_three()

#has to be revised due to the difference between version1.1 and version2.2
def answer_four():
    relationship_df = pd.read_csv('Employee_Relationships.txt', delim_whitespace=True,
                                  header=None, names=['n1', 'n2', 'score'])
    relationship_df = relationship_df.set_index(keys=['n1', 'n2'])
    relationship_df.head()
    G = answer_three()
    result_dict = G.edge
    weight_list = []
    for employee in employee_list:
        part_result = result_dict[employee]
        for key, value in part_result.items():
            new_dict = {"n1":employee,
                        "n2":key,
                        "score2": value['weight']}
            weight_list.append(new_dict)
    interest_df = pd.DataFrame(weight_list)        
    interest_df = interest_df.drop_duplicates()
    interest_df = interest_df.set_index(keys=['n1', 'n2'])
    new_df = relationship_df.merge(interest_df, how='outer', left_index=True, right_index=True)
    new_df['score2'] = new_df['score2'].fillna(0)
    new_df = new_df.dropna()
    return new_df.corr()['score2']['score']

answer3 = answer_four()