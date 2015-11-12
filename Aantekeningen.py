'''

Series:  A series is a one-dimensional NumPy-like array. 
You can put any data type in here, and perform vectorized operations on it. 
A series is also dictionary - like in many ways. Usually this is denoted as "s."
- dictionary-like 

DataFrame: Two-dimensional NumPy-like array. 
Again, any data type can be stuffed in here. 
Usually this is denoted as "df"
- dictionary-like => df['Column1']

index = This is what the data is "associated" by. 
So if you have time series data, like stock price information, generally the "index" is the date

Slicing - Selecting specific batches of data. 
(Move Columns or reorganise, do mathmatical operations on rows)

Referencing prio

'''

#importing AAPL stock data from Yahoo Finance.

import pandas as pd
from pandas import DataFrame

import pandas_datareader.data as web
import datetime
import random
import matplotlib.pyplot as plt

start = datetime.datetime(2010,1,1) 
end = datetime.datetime(2015,5, 9)
AAPL = web.DataReader('AAPL', 'yahoo', start ,end)
AAPL.to_csv('AAPL.csv')
df = pd.read_csv('AAPL.csv', index_col = 'Date', parse_dates=True) #Get csv data with index_colum, and parse dates

def function(data):
	x = random.randrange(0,5)
	return data*x

df['Multiple'] = map(function, df['Close']) #create data frame with column Multiple and map the function here with data from column 'Close'

print(df.head())




"""
df['H-L'] = df.High - df.Low

print df.describe() #describe the whole set with Count, Mean, Std, Min, 25%, 50%, 75%, max

print df.corr() #give the correlation 

print df.cov() #give the covarance

print df[['Volume','H-L']].corr() #correlation between volume on H-L, can also between stocks, peers etc.

df['100MA'] = pd.rolling_mean(df['Close'], 100, min_periods=1) # df['NameofColumn'] = do something cool here


df['Difference'] = df['Close'].diff()

df['STD'] = pd.rolling_std(df['Close'], 25, min_periods=1) #calculating the standard deviation

print df.head

#Plotting on multiple graphs
ax1 = plt.subplot(2,1,1)
df['Close'].plot()

ax2 = plt.subplot(2,1,2, sharex = ax1) #sharex, it shares the axes between both graphs
df['STD'].plot()

plt.show()

df[['Open','High','Low','Close','100MA']].plot()
plt.show()


print(df.head) #print alles met head

df2 = df['Open']

print df2.head()

df3 = df[['Close','Open']] #Here we reference Close and Open for our dataset.

print df3.head()

#del df3['Close']

#print df3.head()

df3.rename(columns={'Close':'CLOSE'}, inplace = True) # This is done with the .rename() function, where you specify what you want to rename in a sort of dictionary

df4 = df3[(df3['CLOSE'] > 100)] #Here we say we just want to see the data that has a close of over 100

print df4

""" 