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
    ax.set_title('{} Regression Results'.format(model), fontsize=15, fontweight='bold')

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

st.subheader("Results")
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

tsmultiplot(y_test, gbm_mean_, 'GBM')

boxplot2(y_test, gbm_mean_, 'GBM')

results_tsplot(qgbm_lower_, gbm_mean_, qgbm_median_, qgbm_upper_,'GBM')



# Mean Features Importance
st.write('GBM mean prediction Features Permutation Importance')
GBM_perm_mean = PermutationImportance(GBM_model, random_state=0).fit(X_test, np.log1p(y_test))
FI_GBM_mean = eli5.show_weights(GBM_perm_mean, feature_names = X_test.columns.tolist())
FI_GBM_mean


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
# Define a function to calculate the permutation feature importance for a given model and data
def permutation_feature_importance(model, X, y):
    # Get the baseline MSE using the original data
    y_pred = model.predict(X)
    baseline_mse = mse(y, y_pred)
    
    # Initialize an empty array to store the feature importances
    feature_importances = np.zeros(X.shape[1])
    
    # Loop over each feature column
    for i in range(X.shape[1]):
        # Make a copy of the original data
        X_permuted = X.copy()
        
        # Shuffle the values of the current feature column
        np.random.shuffle(X_permuted[:, i])
        
        # Get the MSE using the permuted data
        y_pred_permuted = model.predict(X_permuted)
        permuted_mse = mse(y, y_pred_permuted)
        
        # Calculate the feature importance as the difference between the baseline and permuted MSEs
        feature_importances[i] = baseline_mse - permuted_mse
    
    # Return the feature importances array
    return feature_importances

print('GBM mean prediction Features Permutation Importance')
feature_importances = permutation_feature_importance(GBM_model, X_test, np.log1p(y_test))
# Create a pandas dataframe with two columns: Weight and Feature
feature_names = ['place', 'depth', 'longitude', 'latitude']
FI_GBM_perm_mean = pd.DataFrame({'Weight': feature_importances, 'Feature': feature_names})
# Print the feature importances
FI_GBM_perm_mean.sort_values(by='Weight',ascending=False)
FI_GBM_perm_mean




###########################################################
# Mean Features Importance
# Define the values for weight and feature
plt.rcParams['figure.figsize']=(10,10)
weight = [0.0094, 0.2062, 0.0097, -0.0059]
feature = ["place", "depth", "longitude", "latitude"]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot a horizontal bar chart
ax.barh(feature, weight)

# Add labels and title
ax.set_xlabel("Weight", fontsize=15)
ax.set_ylabel("Feature",fontsize=15)
ax.set_title('GBM mean prediction Features Permutation Importance', fontsize=30)
plt.yticks(fontsize=35)
plt.xticks(fontsize=15)
st.pyplot(fig)

# Define the values for weight and feature
plt.rcParams['figure.figsize']=(10,10)
weight = [0.0130, 0.1901 , 0.0043, -0.0286]
feature = ["place", "depth", "longitude", "latitude"]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot a horizontal bar chart
ax.barh(feature, weight)

# Add labels and title
ax.set_xlabel("Weight", fontsize=15)
ax.set_ylabel("Feature",fontsize=15)
ax.set_title('GBM lower prediction Features Permutation Importance', fontsize=30)
plt.yticks(fontsize=35)
plt.xticks(fontsize=15)
st.pyplot(fig)

# Define the values for weight and feature
plt.rcParams['figure.figsize']=(10,10)
weight = [0.0159, 0.1722, -0.0054, -0.0049]
feature = ["place", "depth", "longitude", "latitude"]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot a horizontal bar chart
ax.barh(feature, weight)

# Add labels and title
ax.set_xlabel("Weight", fontsize=15)
ax.set_ylabel("Feature",fontsize=15)
ax.set_title('GBM median prediction Features Permutation Importance', fontsize=30)
plt.yticks(fontsize=35)
plt.xticks(fontsize=15)
st.pyplot(fig)

# Define the values for weight and feature
plt.rcParams['figure.figsize']=(10,10)
weight = [0.0226, 0.0254, 0.0121, -0.0151]
feature = ["place", "depth", "longitude", "latitude"]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot a horizontal bar chart
ax.barh(feature, weight)

# Add labels and title
ax.set_xlabel("Weight", fontsize=15)
ax.set_ylabel("Feature",fontsize=15)
ax.set_title('GBM upper prediction Features Permutation Importance', fontsize=30)
plt.yticks(fontsize=35)
plt.xticks(fontsize=15)
st.pyplot(fig)





