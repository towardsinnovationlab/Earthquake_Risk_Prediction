import numpy as np
import pandas as pd
import streamlit as st


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./data/train.csv')

df_cleaned = pd.read_csv('./data/train_cleaned.csv')

if st.checkbox('Show original data'):
    st.subheader('Raw data')
    st.write(df)

if st.checkbox('Show data used'):
    st.subheader('Raw data')
    st.write(df_cleaned)