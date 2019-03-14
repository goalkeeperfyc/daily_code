#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:05:46 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import networkx as nx

G = nx.MultiGraph()
G.add_node('A',role='manager')
G.add_edge('A','B',relation='friend')
G.add_edge('A','C', relation='business partner')
G.add_edge('A','B', relation='classmate')
#G.node['A']['role'] = 'team member'
#G.node['B']['role'] = 'engineer'

#result = G.edges(['A', 'C'])