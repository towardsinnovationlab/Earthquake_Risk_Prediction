import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
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

df = pd.read_csv('./data/train.csv',index_col=0,parse_dates=True)
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
def is_outlier(x):
    return 'yes' if x == -1 else 'no'

model = IsolationForest(contamination=0.025, random_state=0)
predictions = model.fit_predict(df4[['mag']])
outliers = np.vectorize(is_outlier)(predictions)
outliers_series = pd.DataFrame(outliers, columns=['outliers'], index=df4.index)
df4_ = pd.concat([df4, outliers_series], axis=1)
df5 = df4_.query('outliers=="no"')
# Split data set between target variable and features
X_full = df5.copy()
y = X_full.mag
X_full.drop(['mag'], axis=1, inplace=True)
# Drop useless features
X_full.drop(['year','month','day','hour','outliers'], axis=1, inplace=True)
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [var for var in X_full.columns if
                    X_full[var].dtype == "object"]
# Subset with categorical features
cat = X_full[categorical_cols]
# Select numerical columns
numerical_cols = [var for var in X_full.columns if X_full[var].dtype in ['float64','int64']]
# Subset with numerical features
num = X_full[numerical_cols]
# new dataframe
df_tot = pd.concat([y,num,cat,],axis="columns")
# Remove nan values
df_tot = df_tot.dropna()
# Mean encoding
enc = MeanEncoder()
y = df_tot['mag']
X = df_tot.drop(['mag'], axis=1)
enc.fit(X, y)
X_t = enc.transform(X)
# Test set on 1 year 
split_point = '2009-01-01'
X_train, X_test = X_t[X_t.index<split_point], X_t[X_t.index>=split_point]
y_train, y_test = y[y.index<split_point], y[y.index>=split_point]
len(X_train), len(X_test)
# Split for time series cross validation
ts_cv = TimeSeriesSplit(n_splits=5)
# splits
splits = list(ts_cv.split(X_t, y))

# Evaluation following quantile approach
GBM = GradientBoostingRegressor(random_state=0)
param_grid = {}
GBM_model = GridSearchCV(GBM,param_grid,cv=ts_cv)
GBM_model.fit(X_train, np.log1p(y_train))
gbm_mean_tr = GBM_model.predict(X_train)
gbm_mean_te = GBM_model.predict(X_test)

qGBM = GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=0)
param_grid = {}
qGBM_model_lower = GridSearchCV(qGBM,param_grid,cv=ts_cv)
qGBM_model_lower.fit(X_train, np.log1p(y_train))
qgbm_lower_tr = qGBM_model_lower.predict(X_train)
qgbm_lower_te = qGBM_model_lower.predict(X_test)

qGBM = GradientBoostingRegressor(loss="quantile",alpha=0.5, random_state=0)
param_grid = {}
qGBM_model_median = GridSearchCV(qGBM,param_grid,cv=ts_cv)
qGBM_model_median.fit(X_train, np.log1p(y_train))
qgbm_median_tr = qGBM_model_median.predict(X_train)
qgbm_median_te = qGBM_model_median.predict(X_test)

qGBM = GradientBoostingRegressor(loss='quantile', alpha=0.95,random_state=0)
param_grid = {}
qGBM_model_upper = GridSearchCV(qGBM,param_grid,cv=ts_cv)
qGBM_model_upper.fit(X_train, np.log1p(y_train))
qgbm_upper_tr = qGBM_model_upper.predict(X_train)
qgbm_upper_te = qGBM_model_upper.predict(X_test)

#Train evaluation
mae_gbm_mean_tr = mean_absolute_error(np.log1p(y_train), gbm_mean_tr)
rmse_gbm_mean_tr = np.sqrt(mean_squared_error(np.log1p(y_train), gbm_mean_tr))
mae_qgbm_lower_tr = mean_absolute_error(np.log1p(y_train), qgbm_lower_tr)
rmse_qgbm_lower_tr = np.sqrt(mean_squared_error(np.log1p(y_train), qgbm_lower_tr))
mae_qgbm_median_tr = mean_absolute_error(np.log1p(y_train), qgbm_median_tr)
rmse_qgbm_median_tr = np.sqrt(mean_squared_error(np.log1p(y_train), qgbm_median_tr))
mae_qgbm_upper_tr = mean_absolute_error(np.log1p(y_train), qgbm_upper_tr)
rmse_qgbm_upper_tr = np.sqrt(mean_squared_error(np.log1p(y_train), qgbm_upper_tr))

#Test evaluation
mae_gbm_mean_te = mean_absolute_error(np.log1p(y_test), gbm_mean_te)
rmse_gbm_mean_te = np.sqrt(mean_squared_error(np.log1p(y_test), gbm_mean_te))
mae_qgbm_lower_te = mean_absolute_error(np.log1p(y_test), qgbm_lower_te)
rmse_qgbm_lower_te = np.sqrt(mean_squared_error(np.log1p(y_test), qgbm_lower_te))
mae_qgbm_median_te = mean_absolute_error(np.log1p(y_test), qgbm_median_te)
rmse_qgbm_median_te = np.sqrt(mean_squared_error(np.log1p(y_test), qgbm_median_te))
mae_qgbm_upper_te = mean_absolute_error(np.log1p(y_test), qgbm_upper_te)
rmse_qgbm_upper_te = np.sqrt(mean_squared_error(np.log1p(y_test), qgbm_upper_te))

# GBM metrics table
df_gbm = {'model': ['GBM','GBM'],
        'evaluation':['MAE','RMSE'],
        'train_mean':[mae_gbm_mean_tr,rmse_gbm_mean_tr],
        'test_mean': [mae_gbm_mean_te,rmse_gbm_mean_te],
        'train_lower': [mae_qgbm_lower_tr,rmse_qgbm_lower_tr],
        'test_lower': [mae_qgbm_lower_te,rmse_qgbm_lower_te],
        'train_median': [mae_qgbm_median_tr,rmse_qgbm_median_tr],
        'test_median': [mae_qgbm_median_te,rmse_qgbm_median_te],
        'train_upper': [mae_qgbm_median_tr,rmse_qgbm_upper_tr],
        'test_upper': [mae_qgbm_median_te,rmse_qgbm_upper_te]  
         }
df_gbm = pd.DataFrame(data=df_gbm, columns=['model','evaluation',
                                            'train_lower','test_lower',
                                            'train_mean','test_mean',
                                           'train_median','test_median',
                                           'train_upper','test_upper'])
df_gbm


# Transformations
scaling = MinMaxScaler()
gbm_mean_ = np.expm1(gbm_mean_te).reshape(-1,1)

tsmultiplot(y_test, gbm_mean_, 'GBM')

boxplot2(y_test, gbm_mean_, 'GBM')

