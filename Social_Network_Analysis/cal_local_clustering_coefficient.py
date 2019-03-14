#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:16:28 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import networkx as nx

G = nx.Graph()
G.add_edge('I','J')
G.add_edge('I','K')
G.add_edge('J','K')

lcc_I = nx.clustering(G,'I')
lcc_K = nx.clustering(G,'K')
lcc_J = nx.clustering(G,'J')