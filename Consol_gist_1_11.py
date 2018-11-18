#================== gist_1

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras import initializers
from matplotlib import pyplot
from datetime import datetime
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%matplotlib inline

#================== gist_2

#data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')
#data = pd.read_csv('/home/UFF/IA/Trabalho/Dados/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
data = pd.read_csv('/home/UFF/IA/Trabalho/Dados/Exp10diasVlLow.csv')
data.isnull().values.any()

#================== gist_3

data.head(10)


#================== gist_4

#data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
#group = data.groupby('date')
#Daily_Price = group['Weighted_Price'].mean()
group = data.groupby('DataHora')
Daily_Price = group['VlLow'].mean()
Daily_Price.head()

#================== gist_5

Daily_Price.tail()

#================== gist_6

#from datetime import date
from datetime import datetime

#---- nota: delta.days devolve a diferença em dias excluindo o último dia... 9-1=8 (obvio)
#d0 = date(2016, 1, 1)
#d1 = date(2017, 10, 15)
d0 = datetime(2018, 6, 1, 0, 0, 0) #para extender a 100 dias, usar 2018-03-03
d1 = datetime(2018, 6, 7, 23, 59, 0)
delta = d1 - d0
#days_look = delta.days + 1
days_look = int((delta.days+1)*((delta.seconds/60) + 1))
print(days_look)

#d0 = date(2017, 8, 21)
#d1 = date(2017, 10, 20)
d0 = datetime(2018, 6, 8, 0, 0, 0) #para 10 dias de treinamento , usar 2018-06-02
d1 = datetime(2018, 6, 9, 23, 59, 0)
delta = d1 - d0
#days_from_train = delta.days + 1
days_from_train = int((delta.days+1)*((delta.seconds/60) + 1))
print(days_from_train)

#d0 = date(2017, 10, 15)
#d1 = date(2017, 10, 20)
d0 = datetime(2018, 6, 9, 0, 0, 0)
d1 = datetime(2018, 6, 9, 23, 59, 0)
delta = d1 - d0
#days_from_end = delta.days + 1
days_from_end = int((delta.days+1)*((delta.seconds/60) + 1))
print(days_from_end)

#================== gist_7

df_train= Daily_Price[len(Daily_Price)-days_look-days_from_end:len(Daily_Price)-days_from_train]
df_test= Daily_Price[len(Daily_Price)-days_from_train:]

print(len(df_train), len(df_test))

#================== gist_8

working_data = [df_train, df_test]
working_data = pd.concat(working_data)

working_data = working_data.reset_index()
#working_data['date'] = pd.to_datetime(working_data['date'])
#working_data = working_data.set_index('date')
working_data['DataHora'] = pd.to_datetime(working_data['DataHora'])
working_data = working_data.set_index('DataHora')

#================== gist_9-10

#s = sm.tsa.seasonal_decompose(working_data.Weighted_Price.values, freq=60)
s = sm.tsa.seasonal_decompose(working_data.VlLow.values, freq=1440)

trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
    line = dict(color = ('rgb(244, 146, 65)'), width = 4))
trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
    line = dict(color = ('rgb(66, 244, 155)'), width = 2))

trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
    line = dict(color = ('rgb(209, 244, 66)'), width = 2))

trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
    line = dict(color = ('rgb(66, 134, 244)'), width = 2))

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='seasonal_decomposition')

#================== gist_11

plt.figure(figsize=(15,7))
ax = plt.subplot(211)
#sm.graphics.tsa.plot_acf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
sm.graphics.tsa.plot_acf(working_data.VlLow.values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
#sm.graphics.tsa.plot_pacf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
sm.graphics.tsa.plot_pacf(working_data.VlLow.values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()
