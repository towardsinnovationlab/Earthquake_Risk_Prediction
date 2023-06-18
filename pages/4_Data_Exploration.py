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

if st.checkbox('Show original data'):
    st.write(df)

#if st.checkbox('Show data used'):
#    st.write(df_cleaned)

st.title("Data Exploration")

st.markdown("""
After data cleaning activity the data set is composed by 5 variables and 8.999 rows. 
""")

st.subheader("Summary Statistics")

# formatting index 
df.index = pd.to_datetime(df.index)
# Filter data frmae with homogeneous magnitude type 
df1=df[df['magType']=='mb']
# drop not more helpful variables (type, magType and magSource, because uniques)
df2 = df1.drop(['magType','type','magSource','locationSource'], axis=1)
df3 = df2.copy()
# include date time variables for analysis
df3['year'] = pd.DatetimeIndex(df3.index).year 
# Drop head and tail with inconsistent number of observations 
df3_ = df3.query('year >= 1980 and year <= 2009')
df_cleaned = df3_.drop(['year'], axis=1)

# Formatting features
df_cleaned.index = pd.to_datetime(df_cleaned.index)

# Summarize attribute distributions for data type of variables
st.write('Categorical Variables')
obj_cols = [var for var in df_cleaned.columns if df_cleaned[var].dtype in ['object']]
df_cleaned[obj_cols].describe().T

# Summarize attribute distributions for data type of variables
st.write('Numerical Variables')
num_cols = [var for var in df_cleaned.columns if df_cleaned[var].dtype in ['int64','float64']]
df_cleaned[num_cols].describe().T


df4 = df_cleaned.copy()
# include date time variables for analysis
df4['year'] = pd.DatetimeIndex(df4.index).year 
df4['month'] = pd.DatetimeIndex(df4.index).month
df4['day'] = pd.DatetimeIndex(df4.index).day
df4['hour'] = pd.DatetimeIndex(df4.index).hour

st.write("""Magnitude range is between 3.0 and 6.1, but the scales can reach also 6.5. The highest island with earthquake was Mindanao island and the 
lowest Samar. Looking at the relationship between depth and magnitude there is no linear relationship between magnitude and depth, because it 
changes and can be low both for low magnitude and high magnitude.""")

st.subheader("First ten max magnitude earthquakes")
df_new = df4.copy()
df_new = df_new.sort_values(by='mag', ascending=False)
df_max= df_new.reset_index()
df_max=df_max[['place','mag','depth','year','month','day','hour']].head(10)
df_max

st.subheader("Last ten min magnitude earthquakes")
df_new = df4.copy()
df_new = df_new.sort_values(by='mag', ascending=True)
df_min= df_new.reset_index()
df_min=df_min[['place','mag','depth','year','month','day','hour']].head(10)
df_min

# Vis Chart
def plot_target(data, var):
    fig=plt.figure()
    plt.rcParams['figure.figsize']=(5,5)
    #plt.suptitle('Earthquake Magnitude Exploratory Data Analysis',fontsize=15)
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    st.pyplot(fig)

# Vis Chart
def plot_cat(data, col1):
    plt.rcParams['figure.figsize']=(15,10)
    fig = plt.figure()
    sns.countplot(x=data[col1], data=data).set_title("Barplot {} Variable Distribution".format(col1), fontsize=20)
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

st.plotly_chart(fig, use_container_width=True)

st.subheader("target variable")

st.write("""After removing first year and the last year from the distribution, the outcome seems to assume a shape close to the Normal distribution 
with slight skewness, and the presence of outliers.
From the analysis with past values is clear that there is not the autocorrelation and for this reason no generation of lag features.
""")
plot_target(df4, var='mag')

with sns.plotting_context("paper"):
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=[10, 10])
    plt.suptitle('Earthquake Magnitude Autocorrelation Analysis',fontsize=20)
    for i, ax_ in enumerate(ax.flatten()):
        lag_series = df4["mag"].shift(i + 1)
        pd.plotting.lag_plot(df4["mag"], lag=i + 1, ax=ax_)
        ax_.set_title(f"Lag {i+1}")
        ax_.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.tight_layout()
st.pyplot(fig)    

st.subheader("Categorical variables")

st.write("""The greater observations concern Mindanao Island, that also have registered highest magnitude.
""")
plot_cat(df4, col1='place')

st.subheader("Numerical variables")

st.write("""Numerical features show skewness in the distribution and depth shows many outliers. 
""")

# Select numerical columns
numerical_cols = [var for var in df4.columns if df4[var].dtype in ['float64','int64']]
# Subset with numerical features
num = df4[numerical_cols]
plot_num(num, var='latitude')
plot_num(num, var='longitude')
plot_num(num, var='depth')







