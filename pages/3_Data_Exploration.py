import numpy as np
import pandas as pd
import streamlit as st


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./data/train.csv',index_col=0,parse_dates=True)

df_cleaned = pd.read_csv('./data/train_cleaned.csv',index_col=0,parse_dates=True)

if st.checkbox('Show original data'):
    st.write(df)

if st.checkbox('Show data used'):
    st.write(df_cleaned)
