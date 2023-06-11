import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import datetime
from pandas.plotting import autocorrelation_plot
import scipy
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from feature_engine.timeseries.forecasting import WindowFeatures


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('./data/train.csv',index_col=0,parse_dates=True)


st.title("Time Series Analysis")


# formatting index 
df.index = pd.to_datetime(df.index)
# Filter data frmae with homogeneous magnitude type 
df1=df[df['magType']=='mb']
# drop not more helpful variables (type, magType and magSource, because uniques)
df2 = df1.drop(['magType','type','magSource','locationSource'], axis=1)
df3 = df2.copy()
# include date time variables for analysis
df3['year'] = pd.DatetimeIndex(df3.index).year 
df3['month'] = pd.DatetimeIndex(df3.index).month
df3['day'] = pd.DatetimeIndex(df3.index).day
df3['hour'] = pd.DatetimeIndex(df3.index).hour
# Drop head and tail with inconsistent number of observations 
df4 = df3.query('year >= 1980 and year <= 2009')


def boxplot(data,var):
    plt.rcParams['figure.figsize']=(20,10)
    fig = plt.figure()
    sns.boxplot(x=var, y='mag', data=data, linewidth=5)
    plt.suptitle('Magnitude Distribution per {}'.format(var),fontsize=25)
    plt.xlabel('{}'.format(var), fontsize=15)
    plt.ylabel('mag', fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    st.pyplot(fig)

def tsplot(data,var,period):
    plt.rcParams['figure.figsize']=(20,10)
    fig = plt.figure()
    if period=='all':
        plt.plot(data[var], linewidth=3, color='Orange')
    else:
        plt.plot(data[var].loc[period], linewidth=3,color='Orange') 
    plt.suptitle('Earthquake Magnitude Time Series',fontsize=25)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('{}'.format(var), fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    st.pyplot(fig)

# distribution per year
boxplot(df4,'year')

# Distribution per month
boxplot(df4,'month')

# Distribution per day
boxplot(df4,'day')

# Distribution per hour
boxplot(df4,'hour')

# Look at the whole Magnitude time series
tsplot(df4, var='mag', period='all')


daily_data = df4.groupby(pd.Grouper(freq='D')).agg({'mag': 'mean'})
daily_data_ = daily_data.dropna()

# Additive Decomposition
add_res = seasonal_decompose(x=daily_data_['mag'], model="additive", period=30)
plt.rcParams['figure.figsize']=(20,20)
fig = plt.figure()
add_res.plot()
plt.xlabel('Time', fontsize=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45, fontsize=20)
st.pyplot(fig)

# Multiplicative Decomposition 
mul_res = seasonal_decompose(x=daily_data_['mag'], model="multiplicative", period=30)
plt.rcParams['figure.figsize']=(20,20)
fig = plt.figure()
mul_res.plot()
plt.xlabel('Time', fontsize=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45, fontsize=20)
st.pyplot(fig)

# Create window features for rolling window
transformer = WindowFeatures(
    variables=["mag"],
    functions=["mean", "std"],
    window=[1,7,30,365], # Day, week, month, year.
)

df_rolling = transformer.fit_transform(daily_data_)

# Plot time series with mean rolling window for day, week month and year 
colors = ['orange', 'green', 'red', 'blue']
fig = plt.figure()
df_rolling.filter(
    regex="mag_.*?_mean", # `.*?` means any number of any characters.
    axis=1  # Filter by column names.
).plot(color=colors, linewidth=3,figsize=(20,10))
my_labels=['Mag_window_day', 'Mag_window_week','Mag_window_month','Mag_window_year']
plt.suptitle("Rolling window mean of Earthquake Magnitude",fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('mag', fontsize=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45, fontsize=20)
plt.legend(my_labels,fontsize=20)
st.pyplot(fig)





