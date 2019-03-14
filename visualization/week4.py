#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:30:26 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
basketball_df = pd.read_csv('basketball.csv')
baseball_df = pd.read_csv('baseball.csv')
football_df = pd.read_csv('football.csv')
puck_df = pd.read_csv('puck.csv')

#calculate the wins
basketball_df['per'] = basketball_df['win'] / (basketball_df['win'] + basketball_df['lose']) * 100
baseball_df['per'] = baseball_df['win'] / (baseball_df['win'] + baseball_df['lose']) * 100
football_df['per'] = football_df['win'] / (football_df['win'] + football_df['lose'] + football_df['tie']) * 100
puck_df['per'] = puck_df['win'] / (puck_df['win'] + puck_df['lose'] + puck_df['tie']) * 100

#revise column name
basketball_df = basketball_df.rename(columns={"Season": "year"})

#build a line with y=50
plt.axhline(y=50, zorder=1, color='k')

#set the plot chart
plt.plot(basketball_df['year'], basketball_df['per'], lw=1.5, color='#ff5d9e', label='basketball')
plt.plot(baseball_df['year'], baseball_df['per'], lw=1.5, color='#8f71ff', label='baseball')
plt.plot(football_df['year'], football_df['per'], lw=1.5, color='#82acff', label='football')
plt.plot(puck_df['year'], puck_df['per'], lw=1.5, color='#8bffff', label='puck')

#add label and remove boundary
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Season (year)')
plt.ylabel('Wins (percentage)')
plt.yticks(np.arange(0, 101, step=10))
plt.ylim(0, 100)
plt.title('Miami Four Sports Team Performances since they Founded')
plt.legend(loc=3)
plt.figure()
plt.show()