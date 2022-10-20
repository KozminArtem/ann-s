# Рисует признаки от часа, умеет определять
# среднюю и квантили концентрации в час
# не строит тренировочные кривые
# Исправлена ошибка в y_test, prediction
import matplotlib.pyplot as plt  # plots
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
import seaborn as sns  # more plots
sns.set()
import warnings  
warnings.filterwarnings('ignore')
from itertools import product  # some useful functions
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf  # statistics and econometrics
import statsmodels.tsa.api as smt
from dateutil.relativedelta import relativedelta  # working with dates with style
from scipy.optimize import minimize  # for function minimization
from tqdm.notebook import tqdm
from sklearn.model_selection import TimeSeriesSplit  # you have everything done for you
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8 
import datetime
df = pd.read_csv('../AirQualityUCI/AirQualityUCI.csv', sep = ';') 
features = list(df.columns)
del features[0]
del features[0]
del features[-1]
del features[-1]
for column in features:
    df[column] = [str(x).replace(',','.') for x in df[column]]
    df[column] = df[column].astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S')
df['Date'] = df['Date'].dt.date
df['Time'] = df['Time'].dt.time
df = df.dropna(subset=['Date', 'Time'])
df_new = pd.DataFrame()
df_new['datetime'] = pd.Series([datetime.datetime.combine(df['Date'][i], df['Time'][i]) for i in range(df['Date'].size) if df['Time'][i]])
for feat in features:
    df_new[feat] = df[feat]
                                                                            # GLOBAL VAR
feat_CO = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'T' ,'RH', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)', 'T', 'CO(GT)']
# feat_CO = ['PT08.S2(NMHC)', 'T' , 'CO(GT)']
# feat_CO = ['PT08.S1(CO)', 'CO(GT)']
Degree = 2
# list_delete = []
list_delete = ['T^2','PT08.S2(NMHC)^2']
# list_delete = ['T^2']
T_size = 0.3

feat_target = feat_CO[-1]
l_feat = len(feat_CO) - 1
for feat in feat_CO:
    df_new = df_new[df_new[feat] > -100]
data = pd.DataFrame(df_new[['datetime'] + feat_CO].copy())
print(data.tail(7))
y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mean_s_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
# for time-series cross-validation set 5 folds
tscv = TimeSeriesSplit(n_splits=5)
def timeseries_train_test_split(X, y, test_size):
    if test_size > 1:
        test_index = int(test_size)
        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]
        return X_train, X_test, y_train, y_test        
    if test_size == 1:
        test_index = int(len(X))
        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[:test_index]
        y_test = y.iloc[:test_index]
        return X_train, X_test, y_train, y_test
    else:
        test_index = int(len(X) * (1 - test_size))
        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]
        return X_train, X_test, y_train, y_test

from sklearn.model_selection import train_test_split
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
time_train = pd.DataFrame()
time_test = pd.DataFrame()
temp_str = " "
def plotModelResults(
    model, X_train=X_train, X_test=X_test,string = temp_str, plot_intervals=False, plot_anomalies=False, time_test = time_test, time_train = time_train, y_test = y_test, y_train = y_train
):
    prediction = model.predict(X_test)
    plt.figure(figsize=(15, 7))
    plt.plot(time_test, prediction, label="prediction", marker = 'o',markersize=3, linestyle = 'None', color = "green")
    plt.plot(time_test, y_test.values, label="actual",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
    if plot_intervals:
        cv = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
        )
        mae = cv.mean() * (-1)
        deviation = cv.std()
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        # plt.plot(time_test, lower, "r--", label="upper bond / lower bond", alpha=0.5, )
        # plt.plot(time_test, upper, "r--", alpha=0.5)
        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(time_test, anomalies, "o", markersize=10, linestyle = 'None',label="Anomalies")
    error_mape = mean_absolute_percentage_error(y_test, prediction)
    error_mse = mean_s_error(y_test, prediction)
    plt.title("MAPE: {0:.2f}% ".format(error_mape) + "MSE: {0:.2f} ".format(error_mse) + str(string))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)

def plotCoefficients(model, X_train = X_train):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed");

                                                                            # Polinomial Features

from sklearn.model_selection import (GridSearchCV, StratifiedKFold,cross_val_score)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

List_Train_Size = np.arange(0.01,1.0,0.05)
List_MAPE = np.array([np.NaN] * len(List_Train_Size))
List_MSE = np.array([np.NaN] * len(List_Train_Size))

len_temp = 2
MSE_temp = np.array([np.NaN] * len_temp)
MAPE_temp = np.array([np.NaN] * len_temp)

# for count, t_size in enumerate(List_Train_Size):
#     for i in range(0,len_temp):    
#         # X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 1.0 - t_size)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - t_size)
#         time_train = X_train.dropna()['datetime']
#         time_test = X_test.dropna()['datetime']
#         X_train = X_train.dropna().drop(['datetime'], axis=1)
#         X_test = X_test.dropna().drop(['datetime'], axis=1)
#         poly = PolynomialFeatures(degree=Degree)
#         # X_train_poly = pd.DataFrame(poly.fit_transform(X_train))
#         # X_test_poly = pd.DataFrame(poly.fit_transform(X_test))
#         # list_poly =list(poly.get_feature_names_out())
#         # X_train_poly.set_axis(list_poly, axis = 'columns', inplace=True)
#         # X_test_poly.set_axis(list_poly, axis = 'columns', inplace=True)
#         # X_train_poly = X_train_poly.dropna().drop(list_delete, axis=1)
#         # X_test_poly  = X_test_poly.dropna().drop(list_delete, axis=1)

#         X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
#         X_test_scaled = pd.DataFrame(scaler.transform(X_test))
#         list_X = X_test.columns.values.tolist()
#         X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
#         X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)
#                                                                             # Pol Scaled
#         X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
#         X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
#         list_scaled_poly = list(poly.get_feature_names_out())
#         X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
#         X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
#         X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
#         X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

#         lr = LinearRegression()
#         lr.fit(X_train_scaled_poly, y_train)
#         # lr_cv = cross_val_score(lr, X_train_scaled_poly, y_train, cv=tscv, scoring="neg_mean_absolute_error")
#         # model_cv = GridSearchCV(estimator = lr, scoring= 'r2',cv = folds)      
#         # fit the model
#         # model_cv.fit(X_train, y_train)   
#         prediction = lr.predict(X_test_scaled_poly)
#         # prediction = lr_cv.predict(X_test_scaled_poly)
#         MAPE_temp[i] = mean_absolute_percentage_error(prediction, y_test)
#         MSE_temp[i] = mean_s_error(prediction, y_test)
#     List_MAPE[count] = np.mean(MAPE_temp)
#     List_MSE[count] = np.mean(MSE_temp)
    
# plt.figure(figsize=(15, 7))
# plt.plot(List_Train_Size, List_MAPE, label="training curve MAPE", color = "green")
# plt.legend(loc="best")
# plt.tight_layout()

# plt.figure(figsize=(15, 7))
# plt.plot(List_Train_Size, List_MSE, label="training curve MSE", color = "green")
# plt.legend(loc="best")
# plt.tight_layout()

from sklearn.model_selection import learning_curve

# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 1.0 - t_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']
X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)

poly = PolynomialFeatures(degree=Degree)

# X_train_poly = pd.DataFrame(poly.fit_transform(X_train))
# X_test_poly = pd.DataFrame(poly.fit_transform(X_test))
# list_poly =list(poly.get_feature_names_out())
# X_train_poly.set_axis(list_poly, axis = 'columns', inplace=True)
# X_test_poly.set_axis(list_poly, axis = 'columns', inplace=True)
# X_train_poly = X_train_poly.dropna().drop(list_delete, axis=1)
# X_test_poly  = X_test_poly.dropna().drop(list_delete, axis=1)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))
list_X = X_test.columns.values.tolist()
X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
list_scaled_poly = list(poly.get_feature_names_out())
X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

# train_sizes = List_Train_Size * len(X_train_scaled_poly)
# train_sizes = train_sizes.astype(int)
# print(train_sizes)
# print(len(X_train_scaled_poly))

X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
y_all = y_train.append(y_test, ignore_index=True)

def learning_curves(estimator, data, target, train_sizes, cv, stringg):
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, data, target, train_sizes = train_sizes,
    cv = cv, scoring = 'neg_mean_squared_error')
    # 'neg_mean_squared_error''neg_mean_absolute_percentage_error'
    train_scores_mean = -train_scores.mean(axis = 1)    
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + str(stringg)+' model, cv_par = ' + str(cv)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    # plt.ylim(0,40)

# for cv_par in range(2,30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' no hour ')

# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "sc_pol, degree = " + str(Degree),  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

                                                                # Model Scaled with hour and weekday

data["hour"] = df_new['datetime'].dt.hour
# data["weekday"] = df_new['datetime'].dt.weekday
y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)

# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = T_size)

time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']
X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))
list_X = X_test.columns.values.tolist()
X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

hour_train_scaled = X_train_scaled.dropna()['hour']
hour_test_scaled = X_test_scaled.dropna()['hour']
X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)

poly = PolynomialFeatures(degree=Degree)

X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
list_scaled_poly =list(poly.get_feature_names_out())
X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

X_train_scaled_poly['hour'] = hour_train_scaled
X_test_scaled_poly['hour'] = hour_test_scaled

X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
y_all = y_train.append(y_test, ignore_index=True)

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

                                                                # Pol Scaled with hour and weekday
        
                                                            # LASSO, RIDGE

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import Lasso, Ridge
                                                     # Pol Ridge Scaled with hour and weekday

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(RidgeCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')


#                                                      # Pol LASSO Scaled with hour and weekday

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LassoCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

#                                                                 # BOOSTING
# from xgboost import XGBRegressor
#                                                   # Pol XGBR Scaled with hour and weekday
# # xgb = XGBRegressor(verbosity=0)
# # # xgb.fit(X_train_scaled_poly, y_train)
# # # plotModelResults(xgb,X_train = X_train_scaled_poly,X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol XGBR with h,d", y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(XGBRegressor(verbosity=0), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')


# plt.plot(X_all_scaled_poly['hour'], y_all, label="C(hour)", marker = 'o',markersize=3, linestyle = 'None', color = "green")
    
plt.figure(figsize = (16,10))
sns.boxplot(x = 'hour', y = 'CO(GT)', data = data)
plt.title("Box Plot. CO(hour)")
# plt.savefig('../fig_AQ/Box_CO_hour.png')




print(data)

time_min = 1600
time_max = 2650


plt.figure()
plt.plot(data['datetime'].iloc[time_min:time_max], data['CO(GT)'].iloc[time_min:time_max])




import pywt


sst = data['CO(GT)'].iloc[time_min:time_max]

time = np.arange(time_min,time_max, 1)


print(time)



dt = 1.0


wavelet = 'mexh'
max_scale = 50
min_scale = 0.5

scales = np.arange(min_scale, max_scale, 1)


[cfs, frequencies] = pywt.cwt(sst, scales, wavelet, dt)


period = 1.0/frequencies

print(frequencies)
print(period)

A_scales, B_time = np.meshgrid(time/24, period)


plt.figure('pywt: 2D-график для z = w (a,b)')
plt.title('pywt: Плоскость ab с цветовыми областями ВП', size=12)
plt.contourf(A_scales, B_time, np.abs(cfs), extend='both')
plt.axhline(y = 24, color = 'green', linestyle = '-')
plt.axhline(y = 24*7, color = 'green', linestyle = '-')







# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'PT08.S1(CO)', data = data)
# plt.title("Box Plot. R_CO(hour)")
# # plt.savefig('../fig_AQ/Box_R_CO_hour.png')







# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'T', data = data)

# RH_min = data.RH.min()    
# RH_max = data.RH.max()

# # len_RH = 20
# # RH_list = np.arange(RH_min, RH_max,len_RH)
# data['rh_sort'] = data['RH']/5
# data['rh_sort'] = data['rh_sort'].round() 

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'rh_sort', y = 'CO(GT)', data = data)

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'rh_sort', y = 'PT08.S1(CO)', data = data)

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'rh_sort', y = 'T', data = data)

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'RH', data = data)





# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'PT08.S2(NMHC)', data = data)

# sns.boxplot(x = 'T', y = 'CO(GT)', data = data)

# sns.jointplot(data=data, x='PT08.S1(CO)', y='CO(GT)', kind = 'reg')
# sns.jointplot(data=data, x='PT08.S1(CO)', y='CO(GT)', kind = 'kde')

# sns.jointplot(data=data, x = 'T', y='CO(GT)', kind = 'reg')
# sns.jointplot(data=data, x = 'T', y='CO(GT)', kind = 'kde')

# sns.jointplot(data=data, x = 'PT08.S2(NMHC)', y='CO(GT)', kind = 'reg')
# sns.jointplot(data=data, x = 'PT08.S2(NMHC)', y='CO(GT)', kind = 'kde')

# data_temp = data[data['hour'].isin([4,8,19,23]) ]

# plt.figure(figsize = (16,10))
# sns.scatterplot(data=data_temp, x='PT08.S1(CO)', y='CO(GT)', hue = 'hour', palette='colorblind')

# plt.figure(figsize = (16,10))
# sns.scatterplot(data=data_temp, x='PT08.S2(NMHC)', y='CO(GT)', hue = 'hour', palette='colorblind')

# plt.figure(figsize = (16,10))
# sns.scatterplot(data=data_temp, x='T', y='CO(GT)', hue = 'hour', palette='colorblind')

# MeanCo_hour = np.array([np.NaN] * 24)
# ar_hour = range(24)
# for i in range(24):
#     MeanCo_hour[i] = data[data['hour'] == i]['CO(GT)'].mean()

# data["hour"] = df_new['datetime'].dt.hour
# data["COmean"] = MeanCo_hour[data["hour"]]
# # data["weekday"] = df_new['datetime'].dt.weekday
# y = data.dropna()[feat_target]
# X = data.dropna().drop([feat_target], axis=1)

# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = T_size)

# time_train = X_train.dropna()['datetime']
# time_test = X_test.dropna()['datetime']
# X_train = X_train.dropna().drop(['datetime'], axis=1)
# X_test = X_test.dropna().drop(['datetime'], axis=1)

# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
# X_test_scaled = pd.DataFrame(scaler.transform(X_test))
# list_X = X_test.columns.values.tolist()
# X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
# X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

# hour_train_scaled = X_train_scaled.dropna()['hour']
# hour_test_scaled = X_test_scaled.dropna()['hour']
# X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
# X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)

# COmean_train_scaled = X_train_scaled.dropna()['COmean']
# COmean_test_scaled = X_test_scaled.dropna()['COmean']
# X_train_scaled = X_train_scaled.dropna().drop(['COmean'], axis=1)
# X_test_scaled = X_test_scaled.dropna().drop(['COmean'], axis=1)

# poly = PolynomialFeatures(degree=Degree)

# X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
# X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
# list_scaled_poly =list(poly.get_feature_names_out())
# X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
# X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

# # X_train_scaled_poly['hour'] = hour_train_scaled
# # X_test_scaled_poly['hour'] = hour_test_scaled

# X_train_scaled_poly['COmean'] = COmean_train_scaled 
# X_test_scaled_poly['COmean'] = COmean_test_scaled

# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)






# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "sc_pol with mean hour degree = " + str(Degree),  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

        
# #                                                             # LASSO, RIDGE
# from sklearn.linear_model import LassoCV, RidgeCV
#                                                      # Pol Ridge Scaled with hour and weekday

# ridge = RidgeCV(cv=tscv)
# ridge.fit(X_train_scaled_poly, y_train)
# plotModelResults(ridge, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Ridge with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(ridge, X_train = X_train_scaled_poly)

#                                                      # Pol LASSO Scaled with hour and weekday


# lasso = LassoCV(cv=tscv)
# lasso.fit(X_train_scaled_poly, y_train)
# plotModelResults(lasso, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Lasso with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lasso, X_train = X_train_scaled_poly)






# List_Train_Size = np.arange(0.01,1.0,0.01)

# for cv_par in range(2,30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' no hour ')

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(RidgeCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LassoCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')


# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "sc_pol with mean hour degree = " + str(Degree),  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

        
# #                                                             # LASSO, RIDGE
# from sklearn.linear_model import LassoCV, RidgeCV
#                                                      # Pol Ridge Scaled with hour and weekday

# ridge = RidgeCV(cv=tscv)
# ridge.fit(X_train_scaled_poly, y_train)
# plotModelResults(ridge, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Ridge with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(ridge, X_train = X_train_scaled_poly)

#                                                      # Pol LASSO Scaled with hour and weekday


# lasso = LassoCV(cv=tscv)
# lasso.fit(X_train_scaled_poly, y_train)
# plotModelResults(lasso, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Lasso with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lasso, X_train = X_train_scaled_poly)

# #                                                                 # BOOSTING

# from xgboost import XGBRegressor
#                                                   # Pol XGBR Scaled with hour and weekday

# xgb = XGBRegressor(verbosity=0)
# xgb.fit(X_train_scaled_poly, y_train)

# plotModelResults(xgb,X_train = X_train_scaled_poly,X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol XGBR with h,d", y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)


# CO_hour_Q1 = np.array([np.NaN] * 24)

# for i in range(24):
#     CO_hour_Q1[i] = data[data['hour'] == i]['CO(GT)'].quantile(0.25)

# data["hour"] = df_new['datetime'].dt.hour
# data["COmean"] = CO_hour_Q1[data["hour"]]

# # data["weekday"] = df_new['datetime'].dt.weekday

# y = data.dropna()[feat_target]
# X = data.dropna().drop([feat_target], axis=1)

# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = T_size)

# time_train = X_train.dropna()['datetime']
# time_test = X_test.dropna()['datetime']
# X_train = X_train.dropna().drop(['datetime'], axis=1)
# X_test = X_test.dropna().drop(['datetime'], axis=1)

# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
# X_test_scaled = pd.DataFrame(scaler.transform(X_test))
# list_X = X_test.columns.values.tolist()
# X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
# X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

# hour_train_scaled = X_train_scaled.dropna()['hour']
# hour_test_scaled = X_test_scaled.dropna()['hour']
# X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
# X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)

# COmean_train_scaled = X_train_scaled.dropna()['COmean']
# COmean_test_scaled = X_test_scaled.dropna()['COmean']
# X_train_scaled = X_train_scaled.dropna().drop(['COmean'], axis=1)
# X_test_scaled = X_test_scaled.dropna().drop(['COmean'], axis=1)

# poly = PolynomialFeatures(degree=Degree)

# X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
# X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
# list_scaled_poly =list(poly.get_feature_names_out())
# X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
# X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

# # X_train_scaled_poly['hour'] = hour_train_scaled
# # X_test_scaled_poly['hour'] = hour_test_scaled
# X_train_scaled_poly['COmean'] = COmean_train_scaled 
# X_test_scaled_poly['COmean'] = COmean_test_scaled

# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)

# List_Train_Size = np.arange(0.01,1.0,0.01)

# for cv_par in range(2,30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' no hour ')

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(RidgeCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LassoCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "sc_pol with Q_1 hour degree = " + str(Degree),  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

        
# #                                                             # LASSO, RIDGE
# from sklearn.linear_model import LassoCV, RidgeCV
#                                                      # Pol Ridge Scaled with hour and weekday

# ridge = RidgeCV(cv=tscv)
# ridge.fit(X_train_scaled_poly, y_train)
# plotModelResults(ridge, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Ridge with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(ridge, X_train = X_train_scaled_poly)

#                                                      # Pol LASSO Scaled with hour and weekday

# lasso = LassoCV(cv=tscv)
# lasso.fit(X_train_scaled_poly, y_train)
# plotModelResults(lasso, X_train = X_train_scaled_poly, X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol Lasso with h,d" + str(Degree), y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lasso, X_train = X_train_scaled_poly)

# #                                                                 # BOOSTING
# from xgboost import XGBRegressor
#                                                   # Pol XGBR Scaled with hour and weekday
# xgb = XGBRegressor(verbosity=0)
# xgb.fit(X_train_scaled_poly, y_train)
# plotModelResults(xgb,X_train = X_train_scaled_poly,X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol XGBR with h,d", y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)


plt.show()

