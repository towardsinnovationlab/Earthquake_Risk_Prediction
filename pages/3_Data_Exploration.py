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


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./data/train.csv',index_col=0,parse_dates=True)

df_cleaned = pd.read_csv('./data/train_cleaned.csv',index_col=0,parse_dates=True)

if st.checkbox('Show original data'):
    st.write(df)

if st.checkbox('Show data used'):
    st.write(df_cleaned)

st.title("Data Exploration")

st.markdown("""
The data set is composed by 5 variables and 9.116 rows. 
""")

st.subheader("Summary Statistics")

# Filter data frmae with homogeneous magnitude type 
df1=df[df['magType']=='mb']
# drop not more helpful variables (type, magType and magSource, because uniques)
df2 = df1.drop(['magType','type','magSource','locationSource'], axis=1)
df3 = df2.copy()
df3 = df2.copy()
# include date time variables for analysis
df3['year'] = pd.DatetimeIndex(df3.index).year 
# Drop head and tail with inconsistent number of observations 
df3_ = df3.query('year >= 1980 and year <= 2009')
df_cleaned = df3_.drop(['year'], axis=1)

# Summarize attribute distributions for data type of variables
st.write('Categorical Variables')
obj_cols = [var for var in df_cleaned.columns if df_cleaned[var].dtype in ['object']]
df_cleaned[obj_cols].describe().T

# Summarize attribute distributions for data type of variables
st.write('Numerical Variables')
num_cols = [var for var in df_cleaned.columns if df_cleaned[var].dtype in ['int64','float64']]
df_cleaned[num_cols].describe().T


# Formatting features
df_cleaned.index = pd.to_datetime(df_cleaned.index)

df4 = df_cleaned.copy()
# include date time variables for analysis
df4['year'] = pd.DatetimeIndex(df4.index).year 
df4['month'] = pd.DatetimeIndex(df4.index).month
df4['day'] = pd.DatetimeIndex(df4.index).day
df4['hour'] = pd.DatetimeIndex(df4.index).hour

# Vis Chart
def plot_target(data, var):
    plt.rcParams['figure.figsize']=(20,15)
    fig = plt.figure()
    plt.suptitle('Earthquake Magnitude Exploratory Data Analysis',fontsize=25)
    plt.subplot(2,2,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    plt.subplot(2,2,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(2,2,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    plt.subplot(2,2,4)
    # autocorrelation plot 
    autocorrelation_plot(df[var])
    plt.title("{} vs time autocorrelation".format(var)) 
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    st.pyplot(fig)

# Vis Chart
def plot_cat(data, col1):
    plt.rcParams['figure.figsize']=(20,10)
    fig = plt.figure()
    sns.countplot(x=data[col1], data=data).set_title("Barplot {} Variable Distribution".format(col1),fontsize=20)
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=90, fontsize=15)
    st.pyplot(fig)

# Vis Chart
def plot_num(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    st.pyplot(fig)


fig=px.density_mapbox(df4, lat='latitude',lon='longitude',radius=1,
                    zoom=3.5, mapbox_style='stamen-terrain',center=dict(lat=11,lon=125),
                      title='Earthquake Magnitude Geographical Distribution')
fig.show()


st.subheader("target variable")
plot_target(df4, var='mag')

st.subheader("Categorical variables")
plot_cat(df4, col1='place')

st.subheader("Numerical variables")
# Select numerical columns
numerical_cols = [var for var in df4.columns if df4[var].dtype in ['float64','int64']]
# Subset with numerical features
num = df4[numerical_cols]
plot_num(num, var='latitude')
plot_num(num, var='longitude')
plot_num(num, var='depth')







