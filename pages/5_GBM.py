#setup deterministic results 
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import datetime
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.inspection import PartialDependenceDisplay
from feature_engine.encoding import MeanEncoder
import eli5
from eli5.sklearn import PermutationImportance
import streamlit as st
import pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Gradient Boosting Machine model")


X_test = pd.read_csv('./data/X_test.csv')
#DATA_URL_ytr = ('./data/y_train.csv')
#y_train = pd.read_csv(DATA_URL_ytr)
#DATA_URL_yte = ('./data/y_test.csv')
y_test = pd.read_csv('./data/y_test.csv')


# vis results
def tsmultiplot(data1,data2,var):
    plt.rcParams['figure.figsize']=(20,10)
    data1_ = pd.DataFrame(data1).reset_index()
    data2_ = pd.DataFrame(data2, columns=['mag_pred'])
    data_new = pd.concat([data1_, data2_], axis=1).set_index("time")
    plt.plot(data_new, linewidth=3)
    my_labels=['Mag_Actual', 'Mag_Prediction']
    plt.suptitle('Magnitude Distribution: Actual vs {} Mean Prediction'.format(var),fontsize=25)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Actual vs Prediction',fontsize=20)
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    plt.legend(my_labels,fontsize=15)
    plt.show()

def boxplot2(data1,data2, var):
    plt.rcParams['figure.figsize']=(20,10)
    data1_ = pd.DataFrame(data1).reset_index()
    data2_ = pd.DataFrame(data2, columns=['mag_pred'])
    data_new = pd.concat([data1_, data2_], axis=1).set_index("time")
    #data_new['month'] = pd.to_datetime(data_new.index).month
    data_new['year_month'] = data_new.index.strftime('%Y-%m')
    sns.boxplot(x="year_month", y="value", hue="variable", 
                data=pd.melt(data_new,id_vars="year_month", value_vars=["mag", "mag_pred"]))
    plt.suptitle('Magnitude Distribution per month compared with {} mean prediction'.format(var),fontsize=25)
    plt.xlabel('{}'.format(var), fontsize=15)
    plt.ylabel('value', fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    plt.legend(loc="upper right", fontsize=15)
    plt.show() 

# Model
# loading in the model to predict on the data
with open('GBM_model.pkl', 'rb') as pickle_in:
    GBM_regressor = pickle.load(pickle_in)

# prediction
gbm_mean_te = GBM_regressor.predict(X_test)


# Transformations
scaling = MinMaxScaler()
#qknn_lower_ = scaling.inverse_transform(np.expm1(qknn_lower_te).reshape(-1,1))
#qknn_median_ = scaling.inverse_transform(np.expm1(qknn_median_te).reshape(-1,1))
#qknn_upper_ = scaling.inverse_transform(np.expm1(qknn_upper_te).reshape(-1,1))
gbm_mean_ = np.expm1(gbm_mean_te).reshape(-1,1)

tsmultiplot(y_test, gbm_mean_, 'GBM')

boxplot2(y_test, gbm_mean_, 'GBM')

