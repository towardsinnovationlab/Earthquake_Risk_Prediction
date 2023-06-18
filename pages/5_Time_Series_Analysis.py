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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from feature_engine.timeseries.forecasting import WindowFeatures


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('./data/train.csv',index_col=0,parse_dates=True)


st.title("Time Series Analysis")

st.write("""In the decomposition of the time series, there is no evidence of seasonality patterns, also trend doesn’t show specific up or down 
patterns, but there is just a jump from 1996 to 2000, more clear looking at the rolling window.
The last analysis has been done by looking at the stationarity of the series. From the rolling window mean of observations, 
increasing window sizes have smoothed the shape over time. This allows the model to make use of behaviours seen at different time scales. 
In the end, the Augmented Dickey-Fuller test, which uses an autoregressive model, rejects the null hypothesis which means the time series is 
stationary, it doesn’t have a time-dependent structure.
""")

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
    plt.suptitle('Magnitude Distribution per {}'.format(var),fontsize=30)
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
    plt.suptitle('Earthquake Magnitude Time Series',fontsize=30)
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

st.subheader("Additive Decomposition")

# Additive Decomposition
add_res = seasonal_decompose(x=daily_data_['mag'], model="additive", period=30)
# Plot the components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
plt.xlabel('Time', fontsize=15)
add_res.observed.plot(ax=ax1)
ax1.set_ylabel('Observed',fontsize=15)
add_res.trend.plot(ax=ax2)
ax2.set_ylabel('Trend',fontsize=15)
add_res.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal',fontsize=15)
add_res.resid.plot(ax=ax4)
ax4.set_ylabel('Residual',fontsize=15)
# Display the plot on Streamlit
st.pyplot(fig)

st.subheader("Multiplicative Decomposition")

# Multiplicative Decomposition
add_res = seasonal_decompose(x=daily_data_['mag'], model="multiplicative", period=30)
# Plot the components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
plt.xlabel('Time', fontsize=15)
add_res.observed.plot(ax=ax1)
ax1.set_ylabel('Observed',fontsize=15)
add_res.trend.plot(ax=ax2)
ax2.set_ylabel('Trend',fontsize=15)
add_res.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal',fontsize=15)
add_res.resid.plot(ax=ax4)
ax4.set_ylabel('Residual',fontsize=15)
# Display the plot on Streamlit
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
fig, ax = plt.subplots(figsize=(20,10)) # Create a figure and an axis object
df_rolling.filter(
    regex="mag_.*?_mean", # `.*?` means any number of any characters.
    axis=1  # Filter by column names.
).plot(color=colors, linewidth=3, ax=ax) # Plot on the axis object
my_labels=['Mag_window_day', 'Mag_window_week','Mag_window_month','Mag_window_year']
plt.suptitle("Rolling window mean of Earthquake Magnitude",fontsize=30)
plt.xlabel('Time', fontsize=20)
plt.ylabel('mag', fontsize=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45, fontsize=20)
plt.legend(my_labels,fontsize=20)
st.pyplot(fig) 




