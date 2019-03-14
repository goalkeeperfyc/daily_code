import mplleaflet
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import mplleaflet
get_ipython().run_line_magic('matplotlib', 'widget')

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


#Load de base (locally)
df = pd.read_csv('fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')



#check the first five line of the base
df.head(5)


# In[4]:


#Transform Data value from tenths of degrees C to degree C
df['Data_Value'] = df['Data_Value']*0.1


#Adjusting Dates to pandas datetime
df['Date'] = pd.to_datetime(df['Date'])


#Splitting Year, Month, Day for checking highest and lowest per day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

#Setting new indexes
df = df.set_index(['Month','Day']) 
df.sort_index(inplace = True)


#Selecting the February 29th
df_remove = df.loc[2,29]

#Removing February 29th from the base
df = df[~df.index.isin(df_remove.index)]


#Creating 2 df for max and min temperature for the decade of 2005~2014
dfmax_dec = df[(df['Element'] == 'TMAX') & (df['Year'] >= 2005) & (df['Year'] <= 2014)]
dfmin_dec = df[(df['Element'] == 'TMIN') & (df['Year'] >= 2005) & (df['Year'] <= 2014)]


#Creating 2 data frames for max and min temperatures per day for the decade
dfmax_dec = dfmax_dec.groupby(level=['Month','Day'])['Data_Value'].max()
dfmin_dec = dfmin_dec.groupby(level=['Month','Day'])['Data_Value'].min()



#Creating 2 df for max and min temperature for the the Year of 2015
dfmax_2015 = df[(df['Element'] == 'TMAX') & (df['Year'] == 2015)]
dfmin_2015 = df[(df['Element'] == 'TMIN') & (df['Year'] == 2015)]


#Creating 2 data frames for max and min temperatures per day for the decade
dfmax_2015 = dfmax_2015.groupby(level=['Month','Day'])[['Data_Value','Date']].max()
dfmin_2015 = dfmin_2015.groupby(level=['Month','Day'])[['Data_Value','Date']].min()



#get the 365 days range for the X axis, for convenience i'll use 2015
date_range = df[(df['Year'] == 2015)]['Date'].unique() 


#%matplotlib notebook
fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (10,5) ) 
plt.plot(date_range, dfmax_dec.values, color = '#F67A2F', linewidth=1, alpha = 0.65, label = '2005-2014 Highs')
plt.plot(date_range, dfmin_dec.values, color = '#2567AC', linewidth=1, alpha = 0.65, label = '2005-2014 Lows')
plt.gca().fill_between(date_range
                       ,dfmax_dec
                       ,dfmin_dec
                       ,facecolor='gray'
                       ,alpha=0.05)


# Labels for the legends of the scatter plots below
my_label_max = "2015 Highs"
my_label_min = "2015 Lows"

for idx, rows in dfmax_2015.iterrows():
    if rows['Data_Value'] > dfmax_dec.loc[idx]:
        plt.scatter(rows['Date'], rows['Data_Value'], c = '#E65100', marker = '.', label = my_label_max, alpha=0.7)
        my_label_max = "_nolegend_" # To avoid duplicate labels in the legend
for idx, rows in dfmin_2015.iterrows():
    if rows['Data_Value'] < dfmin_dec.loc[idx]:
        plt.scatter(rows['Date'], rows['Data_Value'], c = '#0091EA', marker = '.', label = my_label_min, alpha=0.7)
        my_label_min = "_nolegend_"

#Define legend
plt.legend(loc=1, frameon=False).get_frame().set_edgecolor('white')

#Define axes
xmin, xmax = date_range[0], date_range[-1]
ax.set_xlim(xmin,xmax)

ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))
# ax.xaxis.set_minor_formatter(ticker.NullFormatter())

xticks = ax.xaxis.get_minor_ticks()
for tick in xticks:
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

#define y axis
margem = 19
if dfmax_dec.max() > dfmax_2015['Data_Value'].max():
    ymax = dfmax_dec.max()+margem
else:
    ymax = dfmax_2015['Data_Value'].max()+margem

if dfmin_dec.min() < dfmin_2015['Data_Value'].min():
    ymin = dfmin_dec.min()-margem
else:
    ymin = dfmin_2015['Data_Value'].min()-margem

ax.set_ylim(ymin,ymax)

#define axis label
ax.set_ylabel('Temperatures in $(^{\circ}$C)')
ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

#Set the Vertical axis grids
ax.yaxis.grid(True, linestyle='-.', linewidth=0.5,color='k', alpha=0.1)

#Define the title
ax.set_title('Ann Arbor, Michigan Record Temperatures (2005-2015)')