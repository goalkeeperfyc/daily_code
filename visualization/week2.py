
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

df = pd.read_csv('fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')

df['temperature'] = df['Data_Value'] / 10

df['datetime'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

df['month-day'] = df['datetime'].dt.strftime('%m/%d')
df['year'] = df['datetime'].dt.year

df_2015 = df[df['year'] == 2015]
df_previous2015 = df[df['year'] < 2015]

agg_2015 = df_2015.groupby('month-day')['temperature'].agg(['max', 'min'])
agg_previous2015 = df_previous2015.groupby('month-day')['temperature'].agg(['max', 'min'])
    
agg_previous2015 = agg_previous2015.drop(['02/29'])

agg_2015 = agg_2015.reset_index()
agg_previous2015 = agg_previous2015.reset_index()

agg_2015 = agg_2015.rename(columns={'min': 'min_2015', 'max': 'max_2015'})
agg_previous2015 = agg_previous2015.rename(columns={'min': 'min_pre', 'max': 'max_pre'})

new_df = agg_2015.merge(agg_previous2015, left_on='month-day', right_on='month-day')

new_df['lower'] = new_df['min_2015'] < new_df['min_pre']
new_df['higher'] = new_df['max_2015'] > new_df['max_pre']

df_2015 = new_df[(new_df['lower'] == True) | (new_df['higher'] == True)]

plt.plot(agg_previous2015['month-day'], agg_previous2015['min_pre'], color='b', lw=0.5, label='lowest through 2005-2014')
plt.plot(agg_previous2015['month-day'], agg_previous2015['max_pre'], color='r', lw=0.5, label='highest through 2005-2014')
plt.scatter(df_2015['month-day'], df_2015['min_2015'], color='m', marker='o', s=30, label='record low achieved in 2015')
plt.scatter(df_2015['month-day'], df_2015['max_2015'], color='k', marker='o', s=30, label='record high achieved in 2015')

ax = plt.gca()
ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
ax.xaxis.set_major_locator(dates.MonthLocator())
loc, labels = plt.xticks()
plt.setp(labels, rotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.subplots_adjust(bottom=0.15)
plt.legend(loc='best')
plt.xlabel('TimeLine')
plt.ylabel('Temperature (Deg C)')
plt.title('Temperature Recorded each Day')

plt.gca().fill_between(agg_previous2015['month-day'], 
                       agg_previous2015['min_pre'], 
                       agg_previous2015['max_pre'], 
                       facecolor='blue', 
                       alpha=0.25)

plt.show()
