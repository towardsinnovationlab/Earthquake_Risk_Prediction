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
import streamlit as st
import eli5
from eli5.sklearn import PermutationImportance


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Gradient Boosting Machine model")

# vis results
def tsmultiplot(data1,data2,var):
    plt.rcParams['figure.figsize']=(20,10)
    fig=plt.figure()
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
    st.pyplot(fig)

def boxplot2(data1,data2, var):
    plt.rcParams['figure.figsize']=(20,10)
    fig=plt.figure()
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
    st.pyplot(fig)
    
def results_tsplot(results_lower, results_mean, results_median, results_upper,model):
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(18, 10))


    # Plot the actual data points
    ax.plot(X_test.index, y_test, color='blue', label='Actual')

    # Plot the predicted mean
    ax.plot(X_test.index, results_median, color='red', linestyle='dashed', linewidth=3,label='Predicted Median')
    #ax.plot(X_test.index, results_mean, color='yellow', linestyle='dotted', linewidth=3,label='Predicted Mean')


    # Plot the lower and upper quantiles
    results_lower_flat = np.ravel(results_lower)
    results_upper_flat = np.ravel(results_upper)
    ax.fill_between(X_test.index, results_lower_flat, results_upper_flat, alpha=0.4, color='green', 
                linewidth=3,label='Prediction Interval')

    # Set the x-axis label and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('{} Regression Results'.format(model), fontsize=25, fontweight='bold')

    # Set the tick label font size
    ax.tick_params(axis='both', labelsize=12)

    # Add a legend
    ax.legend(fontsize=12)

    # Show the chart
    st.pyplot(fig)

def train_metrics_model(mae_lower, mae_mean, mae_median, mae_upper, 
                        rmse_lower, rmse_mean, rmse_median, rmse_upper, model):
    plt.rcParams['figure.figsize']=(15,10)
    # Create a dictionary of the data for MAE
    mae_train_data = {
    'train_lower': mae_lower,
    'train_mean': mae_mean,
    'train_median': mae_median,
    'train_upper': mae_upper
    }

    # Create a dictionary of the data for RMSE
    rmse_train_data = {
    'train_lower': rmse_lower,
    'train_mean': rmse_mean,
    'train_median': rmse_median,
    'train_upper': rmse_upper
    }

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the MAE data as a red dashed line with asterisk markers
    ax.plot([x.replace('_', ' ').title() for x in mae_train_data.keys()], list(mae_train_data.values()), color='red', linestyle='--', linewidth=5, marker='*', label='MAE')

    # Plot the RMSE data as a blue solid line with triangle markers
    ax.plot([x.replace('_', ' ').title() for x in rmse_train_data.keys()], list(rmse_train_data.values()), color='blue', linestyle='-', linewidth=5, marker='^', label='RMSE')

    # Set the title and axis labels
    ax.set_title('Train Metrics for {}'.format(model), fontsize=30)
    ax.set_xlabel('Metric', fontsize=30)
    ax.set_ylabel('Value', fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    # Add a legend
    ax.legend(fontsize=15)

    # Show the plot
    st.pyplot(fig)
    
def test_metrics_model(mae_lower, mae_mean, mae_median, mae_upper, 
                        rmse_lower, rmse_mean, rmse_median, rmse_upper, model):
    plt.rcParams['figure.figsize']=(15,10)
    # Create a dictionary of the data for MAE
    mae_test_data = {
    'test_lower': mae_lower,
    'test_mean': mae_mean,
    'test_median': mae_median,
    'test_upper': mae_upper
    }

    # Create a dictionary of the data for RMSE
    rmse_test_data = {
    'test_lower': rmse_lower,
    'test_mean': rmse_mean,
    'test_median': rmse_median,
    'test_upper': rmse_upper
    }

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the MAE data as a red dashed line with asterisk markers
    ax.plot([x.replace('_', ' ').title() for x in mae_test_data.keys()], list(mae_test_data.values()), color='red', linestyle='--', linewidth=5, marker='*', label='MAE')

    # Plot the RMSE data as a blue solid line with triangle markers
    ax.plot([x.replace('_', ' ').title() for x in rmse_test_data.keys()], list(rmse_test_data.values()), color='blue', linestyle='-', linewidth=5, marker='^', label='RMSE')

    # Set the title and axis labels
    ax.set_title('Test Metrics for {}'.format(model), fontsize=30)
    ax.set_xlabel('Metric', fontsize=30)
    ax.set_ylabel('Value', fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    # Add a legend
    ax.legend(fontsize=15)

    # Show the plot
    st.pyplot(fig)
    


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
# Split for time series cross validation
ts_cv = TimeSeriesSplit(n_splits=5)
# splits
splits = list(ts_cv.split(X_t, y))

# Modelling 
GBM = GradientBoostingRegressor(random_state=0)
param_grid = {}
GBM_model = GridSearchCV(GBM,param_grid,cv=ts_cv)
GBM_model.fit(X_train, np.log1p(y_train))
gbm_mean_tr = GBM_model.predict(X_train)
gbm_mean_te = GBM_model.predict(X_test)

# Modelling with quantile approach
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

st.subheader("Modelling")

st.write("""The model has fitted in the noraml way with the mean regression and then with 3 quantile points: 0.05 lower quantile, 
0.50 median quantile, and 0.95 upper quantile.
The dataset was splitted in two parts: train set from 1980 to 2008 and test set equal to 2009 as test set used for prediction.
In the training process has been used the time series split cross validation with 5 folds suitable for a proper evaluation splitting strategies 
that takes into account the temporal structure of the dataset to evaluate the model's ability to predict data points in the future.
Gradient Boosting Machine in scikit-learn allows to model the quantile loss function, just declare it and so on the quantile point.
Models Have been trained with default hyperparameters values, given several points to train. Default values are enough to understand 
if the model is suitable for your data points. 
""")

st.subheader("Results")

st.write("""Are reported evaluation results for train and test set with mean prediction, lower quantile=0.05, median quantile=0.50, 
upper quantile=0.95.
Results are coming from the angles and lines are just drafted for a better visualization.
""")
df_gbm

# create data as a list of dictionaries
df_gbm_MAE_tr = [
  {'MAE': 'train_lower', 'value': mae_qgbm_lower_tr},
  {'MAE': 'train_mean', 'value': mae_gbm_mean_tr},
  {'MAE': 'train_median', 'value': mae_qgbm_median_tr},
  {'MAE': 'train_upper', 'value': mae_qgbm_upper_tr},
]

# create the DataFrame
df_gbm_MAE_tr = pd.DataFrame(df_gbm_MAE_tr)
# create data as a list of dictionaries
df_gbm_RMSE_tr = [
  {'RMSE': 'train_lower', 'value': rmse_qgbm_lower_tr},
  {'RMSE': 'train_mean', 'value': rmse_gbm_mean_tr},
  {'RMSE': 'train_median', 'value': rmse_qgbm_median_tr},
  {'RMSE': 'train_upper', 'value': rmse_qgbm_upper_tr},
]

# create the DataFrame
df_gbm_RMSE_tr = pd.DataFrame(df_gbm_RMSE_tr)
# create data as a list of dictionaries
df_gbm_MAE_te = [
  {'MAE': 'test_lower', 'value': mae_qgbm_lower_te},
  {'MAE': 'test_mean', 'value': mae_gbm_mean_te},
  {'MAE': 'test_median', 'value': mae_qgbm_median_te},
  {'MAE': 'test_upper', 'value': mae_qgbm_upper_te},
]

# create the DataFrame
df_gbm_MAE_te = pd.DataFrame(df_gbm_MAE_te)
# create data as a list of dictionaries
df_gbm_RMSE_te = [
  {'RMSE': 'test_lower', 'value': rmse_qgbm_lower_te},
  {'RMSE': 'test_mean', 'value': rmse_gbm_mean_te},
  {'RMSE': 'test_median', 'value': rmse_qgbm_median_te},
  {'RMSE': 'test_upper', 'value': rmse_qgbm_upper_te},
]

# create the DataFrame
df_gbm_RMSE_te = pd.DataFrame(df_gbm_RMSE_te)

train_metrics_model(mae_qgbm_lower_tr, mae_gbm_mean_tr, mae_qgbm_median_tr, mae_qgbm_upper_tr,
                   rmse_qgbm_lower_tr, rmse_gbm_mean_tr, rmse_qgbm_median_tr, rmse_qgbm_upper_tr,'GBM')

test_metrics_model(mae_qgbm_lower_te, mae_gbm_mean_te, mae_qgbm_median_te, mae_qgbm_upper_te,
                   rmse_qgbm_lower_te, rmse_gbm_mean_te, rmse_qgbm_median_te, rmse_qgbm_upper_te,'GBM')

# Transformations
scaling = MinMaxScaler()
qgbm_lower_ = np.expm1(qgbm_lower_te).reshape(-1,1)
qgbm_median_ = np.expm1(qgbm_median_te).reshape(-1,1)
qgbm_upper_ = np.expm1(qgbm_upper_te).reshape(-1,1)
gbm_mean_ = np.expm1(gbm_mean_te).reshape(-1,1)

st.write("""In the following charts are reported the projection of results in 2009, firstly comparing actual values with mean regression prediction, 
and then actual values with quantile regression. The red line represents the median quantile and with green color is reported the prediction
interval. The median quantile generalize well the observations and the prediction interval is large, able to handle spikes.
""")

tsmultiplot(y_test, gbm_mean_, 'GBM')

boxplot2(y_test, gbm_mean_, 'GBM')

results_tsplot(qgbm_lower_, gbm_mean_, qgbm_median_, qgbm_upper_,'GBM')

st.subheader("""Quantile Calibration Assessment""")

st.markdown("""How can we trust about performance metrics results?
The first analysis is to look if the predicted quantiles match the observed quantiles for each quantile point.
The model seems well calibrated for the upper quantile and the median quantile, instead seems to be weak for the lower quantile.
At the end gradient boosting is able to capture almost all of the true values within its prediction interval.""")

# Lower Quantile
st.write("""Lower Quantile - > bad""")
quantile_GBM_lower=(qgbm_lower_ > y_test.values).mean()
quantile_GBM_lower
# Median Quantile
st.write("""Median Quantile - > good""")
percentile_GBM_median=(qgbm_median_ > y_test.values).mean()
percentile_GBM_median
# Upper Quantile
st.write("""Upper Quantile - > good""")
quantile_GBM_upper=(qgbm_upper_ > y_test.values).mean()
quantile_GBM_upper
# Coverage
st.write("""Prediction Interval Coverage - > good""")
coverage_GBM = np.logical_and(
               qgbm_lower_ < y_test.values,
               qgbm_upper_ > y_test.values).mean()
coverage_GBM


# Mean Features Importance
#st.write('GBM mean prediction Features Permutation Importance')
GBM_perm_mean = PermutationImportance(GBM_model, random_state=0).fit(X_test, np.log1p(y_test))
# Create a dataframe with feature importances and names
FI_GBM_mean = pd.DataFrame(dict(
    feature_names=X_test.columns.tolist(),
    feature_importance=GBM_perm_mean.feature_importances_,
    std=GBM_perm_mean.feature_importances_std_,
))
# Create a horizontal barplot with feature importances
plt.rcParams['figure.figsize']=(8,8)
fig=plt.figure()
plt.barh(y=FI_GBM_mean["feature_names"], width=FI_GBM_mean["feature_importance"], color="red")
plt.title("GBM mean prediction Features Permutation Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
# Display the chart with streamlit
st.pyplot(fig)

st.subheader("""Feature Importance""")

st.write("""In the charts are reported the permutation feature importance and we can see that depth has the degree of importance 
with the highest value in several points, seems to be coherent in every prediction.
""")

# Lower Features Importance
#st.write('GBM lower prediction Features Permutation Importance')
GBM_perm_lower = PermutationImportance(qGBM_model_lower, random_state=0).fit(X_test, np.log1p(y_test))
# Create a dataframe with feature importances and names
FI_GBM_lower = pd.DataFrame(dict(
    feature_names=X_test.columns.tolist(),
    feature_importance=GBM_perm_lower.feature_importances_,
    std=GBM_perm_lower.feature_importances_std_,
))
# Create a horizontal barplot with feature importances
plt.rcParams['figure.figsize']=(8,8)
fig=plt.figure()
plt.barh(y=FI_GBM_lower["feature_names"], width=FI_GBM_lower["feature_importance"], color="red")
plt.title("GBM lower prediction Features Permutation Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
# Display the chart with streamlit
st.pyplot(fig)

# median Features Importance
#st.write('GBM median prediction Features Permutation Importance')
GBM_perm_median = PermutationImportance(qGBM_model_median, random_state=0).fit(X_test, np.log1p(y_test))
# Create a dataframe with feature importances and names
FI_GBM_median = pd.DataFrame(dict(
    feature_names=X_test.columns.tolist(),
    feature_importance=GBM_perm_median.feature_importances_,
    std=GBM_perm_median.feature_importances_std_,
))
# Create a horizontal barplot with feature importances
plt.rcParams['figure.figsize']=(8,8)
fig=plt.figure()
plt.barh(y=FI_GBM_median["feature_names"], width=FI_GBM_median["feature_importance"], color="red")
plt.title("GBM median prediction Features Permutation Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
# Display the chart with streamlit
st.pyplot(fig)

# upper Features Importance
#st.write('GBM upper prediction Features Permutation Importance')
GBM_perm_upper = PermutationImportance(qGBM_model_upper, random_state=0).fit(X_test, np.log1p(y_test))
# Create a dataframe with feature importances and names
FI_GBM_upper = pd.DataFrame(dict(
    feature_names=X_test.columns.tolist(),
    feature_importance=GBM_perm_upper.feature_importances_,
    std=GBM_perm_upper.feature_importances_std_,
))
# Create a horizontal barplot with feature importances
plt.rcParams['figure.figsize']=(8,8)
fig=plt.figure()
plt.barh(y=FI_GBM_upper["feature_names"], width=FI_GBM_upper["feature_importance"], color="red")
plt.title("GBM upper prediction Features Permutation Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
# Display the chart with streamlit
st.pyplot(fig)

st.subheader("""Partial Dependence Plot""")

st.write("""Depth variable that is the most relevant, loooking in the Partial Dependence Plot it changes pattern with the upper quantile""")
# Mean Partial Dependence Plot
plt.rcParams['figure.figsize']=(25,15)
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.suptitle('Partial dependence of Earthquake Magnitude with GBM model',fontsize=30)

features = ['latitude','longitude','depth','place']
for i, feature in enumerate(features):
    ax = axs.flatten()[i]
    PartialDependenceDisplay.from_estimator(GBM_model, X_t, [feature], ax=ax, kind='average', random_state=0)
    ax.set_title(f'PDP of Earthquake Magnitude on {feature} with GBM model', fontsize=20)
st.pyplot(fig)

# Lower Partial Dependence Plot
plt.rcParams['figure.figsize']=(25,15)
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.suptitle('Partial dependence of Earthquake Magnitude with GBM lower quantile',fontsize=30)

features = ['latitude','longitude','depth','place']
for i, feature in enumerate(features):
    ax = axs.flatten()[i]
    PartialDependenceDisplay.from_estimator(qGBM_model_lower, X_t, [feature], ax=ax, kind='average', random_state=0)
    ax.set_title(f'PDP of Earthquake Magnitude on {feature} with GBM quantile', fontsize=20)
st.pyplot(fig)

# Median Partial Dependence Plot
plt.rcParams['figure.figsize']=(25,15)
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.suptitle('Partial dependence of Earthquake Magnitude with GBM median quantile',fontsize=30)

features = ['latitude','longitude','depth','place']
for i, feature in enumerate(features):
    ax = axs.flatten()[i]
    PartialDependenceDisplay.from_estimator(qGBM_model_median, X_t, [feature], ax=ax, kind='average', random_state=0)
    ax.set_title(f'PDP of Earthquake Magnitude on {feature} with GBM quantile', fontsize=20)
st.pyplot(fig)

# Upper Partial Dependence Plot
plt.rcParams['figure.figsize']=(25,15)
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.suptitle('Partial dependence of Earthquake Magnitude with GBM upper quantile',fontsize=30)

features = ['latitude','longitude','depth','place']
for i, feature in enumerate(features):
    ax = axs.flatten()[i]
    PartialDependenceDisplay.from_estimator(qGBM_model_upper, X_t, [feature], ax=ax, kind='average', random_state=0)
    ax.set_title(f'PDP of Earthquake Magnitude on {feature} with GBM quantile', fontsize=20)
st.pyplot(fig)


