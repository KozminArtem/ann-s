# Построение тренировочных кривых в полиномиальных 
# моделях на всех данных при разных параметрах кросс-
# валидации cv_par с часами и без.
# В конце постройка ящичных диаграмм и диаграмм рассеяния
# Исправлена ошибка y_test, prediction

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

print("dataframe")
print(df)
print("dataframe")

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

feat_CO = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'T' , 'CO(GT)']

# feat_CO = ['PT08.S1(CO)', 'PT08.S2(NMHC)' , 'CO(GT)']

# feat_CO = ['PT08.S1(CO)', 'CO(GT)']

# feat_CO = ['PT08.S2(NMHC)', 'T' , 'CO(GT)']
# feat_CO = ['PT08.S1(CO)', 'CO(GT)']
Degree = 2
list_delete = []
# list_delete = ['T^2','PT08.S2(NMHC)^2']
# list_delete = ['T^2','R_NM^2']


T_size = 2000

feat_target = feat_CO[-1]
l_feat = len(feat_CO) - 1

for feat in feat_CO:
    df_new = df_new[df_new[feat] > -100]

print("dataframe new")
print(df_new)
print("dataframe new")


data = pd.DataFrame(df_new[['datetime'] + feat_CO].copy())
# print(data.tail(7))
# print("AAAAAAAAA")
# print(data)

# print(data.iloc[:500].describe())

# print(data.iloc[500:1000].describe())


# g = sns.pairplot(data.iloc[1000:1500], kind = 'scatter')  
# g.map_lower(sns.kdeplot, levels=4, color=".2")

# print(data.iloc[1000:1500].describe())


# print(data.iloc[1500:2000].describe())


# g1 = sns.pairplot(data.iloc[1500:2000], kind = 'scatter')  
# g1.map_lower(sns.kdeplot, levels=4, color=".2")


# print(data.iloc[:2304])

# print(data.iloc[2304:])

y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)



data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mean_s_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100



def gate_rate_error(y_true,y_pred):
    gre = 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    len_y_true = len(y_true)
    for j in range(len_y_true):
        if abs(y_true[j] - y_pred[j]) > 0.25 * abs(y_true[j]):
            gre = gre + 1
    return (gre/len_y_true)*100    


def arr_gate_rate_error(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    len_y_true = len(y_true)
    array_gre = np.array([np.NaN] * len_y_true)

    for j in range(len_y_true):
        if abs(y_true[j] - y_pred[j]) > 0.25 * abs(y_true[j]):
            array_gre[j] = 1.0
        else: 
            array_gre[j] = 0.0   
    return array_gre        


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
    model, X_train=X_train, X_test=X_test,string = temp_str, plot_intervals=False, plot_anomalies=False,\
    plot_diff = False, time_test = time_test, time_train = time_train, y_test = y_test, y_train = y_train
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
    print("MSE",error_mse)
    error_gre = gate_rate_error(y_test, prediction)
    plt.title("MAPE: {0:.2f}% ".format(error_mape) + "MSE: {0:.2f} ".format(error_mse) + "GRE: {0:.2f}%".format(error_gre) + str(string))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    

    error_mape = mean_absolute_percentage_error(y_test.values,prediction)

    array_diff = abs(y_test.values - prediction)
    array_mse  = (y_test.values - prediction)**2
    # print(type(array_mse), len(array_mse))
    array_mape = (array_diff/y_test.values) * 100
    array_gre  = arr_gate_rate_error(y_test.values,prediction)

    # print(array_mse.mean())
    # print(array_mse.max())
    # print(array_mse.min())
    # print(np.sort(array_mse))

    error_mse = mean_s_error(y_test.values,prediction)
    print("MSE VALUES",error_mse)
    error_gre = gate_rate_error(y_test.values,prediction)

    


    temp = pd.DataFrame()
    temp['CO_true'] = y_test.values
    temp['MSE']  = array_mse
    temp['MAPE'] = array_mape
    temp['GRE']  = array_gre
    temp['T']    = X_test['T']
    temp['R_CO'] = X_test['R_CO']
    temp['R_NM'] = X_test['R_NM']
    temp['Pred'] = prediction
    # temp['hour'] = X_test['hour']
    # temp['COmean'] = X_test['COmean']
    print(len(time_test.dt.month))
    print(len(prediction))
    print(type(time_test.dt.month))
    print(type(prediction))


    temp['Month'] = np.array(time_test.dt.month)
    print(temp['Month'])
    print(temp['GRE'].mean())


    if plot_diff:
        # plt.figure(figsize=(15, 7))
        # plt.plot(time_test, array_diff, label="dif", marker = 'o',markersize=3, linestyle = 'None', color = "green")
        # plt.title("Diff Mean: {0:.2f} ".format(array_diff.mean()) + "Min: {0:.2f} ".format(array_diff.min()) + "Max: {0:.2f} ".format(array_diff.max()) + str(string))
        # plt.legend(loc="best")
        # plt.tight_layout()
        # plt.grid(True)
        

        plt.figure(figsize=(15, 7))
        plt.plot(np.linspace(0, 10),np.linspace(0, 10)) 

        plt.plot(y_test.values, prediction, label="Pred(True)",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.plot(np.linspace(0, 10),1.25*np.linspace(0, 10), color = "red")
        plt.plot(np.linspace(0, 10),0.75*np.linspace(0, 10), color = "red")

        plt.title("Pred(True)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)
        # plt.savefig('fig_LinPol/Pred(True)' + str(string) +'.png')



        # plt.figure(figsize = (16,10))
        # sns.boxplot(x = 'CO_true', y = 'Pred', data = temp)
        # plt.plot(np.linspace(0, 10),np.linspace(0, 10))





        plt.figure(figsize=(15, 7))
        plt.plot(time_test, array_mse, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("MSE Mean: {0:.2f} ".format(array_mse.mean()) + "Min: {0:.2f} ".format(array_mse.min()) + "Max: {0:.2f} ".format(array_mse.max()) + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(y_test.values, array_mse, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("Mse(y_true)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize = (16,10))
        sns.boxplot(x = 'CO_true', y = 'MSE', data = temp)
        # plt.savefig('fig_LinPol/Box_MSE' + str(string) +'.png')



        plt.figure(figsize=(15, 7))
        plt.plot(temp['T'], array_mse, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("Mse(T)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(temp['R_CO'], array_mse, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("Mse(R_CO)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(temp['R_NM'], array_mse, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("Mse(R_NM)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize = (16,10))
        sns.boxplot(x = 'Month', y = 'MSE', data = temp)












        plt.figure(figsize=(15, 7))
        plt.plot(time_test, array_mape, label="MAPE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("MAPE Mean: {0:.2f} ".format(array_mape.mean()) + "Min: {0:.2f} ".format(array_mape.min()) + "Max: {0:.2f} ".format(array_mape.max()) + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(y_test.values, array_mape, label="MAPE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("Mape(y_true)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize = (16,10))
        sns.boxplot(x = 'CO_true', y = 'MAPE', data = temp)
        # plt.savefig('fig_LinPol/Box_MAPE' + str(string) +'.png')


        plt.figure(figsize=(15, 7))
        plt.plot(temp['T'], array_mape, label="MAPE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("MAPE(T)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(temp['R_CO'], array_mape, label="MAPE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("MAPE(R_CO)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(temp['R_NM'], array_mape, label="MAPE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("MAPE(R_NM)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize = (16,10))
        sns.boxplot(x = 'Month', y = 'MAPE', data = temp)




        # plt.figure(figsize=(15, 7))
        # plt.plot(temp['hour'], array_mape, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        # plt.title("MAPE(hour)" + str(string))
        # plt.legend(loc="best")
        # plt.tight_layout()
        # plt.grid(True)

        # plt.figure(figsize = (16,10))
        # sns.boxplot(x = 'COmean', y = 'MAPE', data = temp)






        plt.figure(figsize=(15, 7))
        plt.plot(time_test, array_gre, label="GRE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("GRE(time_test)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        plt.figure(figsize=(15, 7))
        plt.plot(y_test.values, array_gre, label="GRE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        plt.title("GRE(y_true)" + str(string))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)

        # plt.figure(figsize = (16,10))
        # sns.boxplot(x = 'Month', y = 'GRE', data = temp)


        plt.figure(figsize = (16,10))
        for mon in range(1,13):
            plt.plot(mon, temp[temp['Month'] == mon]['GRE'].mean())



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

# List_Train_Size = np.arange(0.01,1.0,0.02)
# List_MAPE = np.array([np.NaN] * len(List_Train_Size))
# List_MSE = np.array([np.NaN] * len(List_Train_Size))
# len_temp = 2
# MSE_temp = np.array([np.NaN] * len_temp)
# MAPE_temp = np.array([np.NaN] * len_temp)

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

# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 2000)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)

# time_train = X_train.dropna()['datetime']
# time_test = X_test.dropna()['datetime']
# X_train = X_train.dropna().drop(['datetime'], axis=1)
# X_test = X_test.dropna().drop(['datetime'], axis=1)
# poly = PolynomialFeatures(degree=Degree)
# # X_train_poly = pd.DataFrame(poly.fit_transform(X_train))
# # X_test_poly = pd.DataFrame(poly.fit_transform(X_test))
# # list_poly =list(poly.get_feature_names_out())
# # X_train_poly.set_axis(list_poly, axis = 'columns', inplace=True)
# # X_test_poly.set_axis(list_poly, axis = 'columns', inplace=True)
# # X_train_poly = X_train_poly.dropna().drop(list_delete, axis=1)
# # X_test_poly  = X_test_poly.dropna().drop(list_delete, axis=1)
# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
# X_test_scaled = pd.DataFrame(scaler.transform(X_test))
# list_X = X_test.columns.values.tolist()
# X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
# X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)
# X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
# X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
# list_scaled_poly = list(poly.get_feature_names_out())
# X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
# X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
# X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)


# # train_sizes = List_Train_Size * len(X_train_scaled_poly)
# # train_sizes = train_sizes.astype(int)
# # print(train_sizes)
# # print(len(X_train_scaled_poly))
# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)
# def learning_curves(estimator, data, target, train_sizes, cv, stringg):
#     train_sizes, train_scores, validation_scores = learning_curve(
#     estimator, data, target, train_sizes = train_sizes,
#     cv = cv, scoring = 'neg_mean_squared_error')
#     # 'neg_mean_squared_error''neg_mean_absolute_percentage_error'
#     train_scores_mean = -train_scores.mean(axis = 1)
#     validation_scores_mean = -validation_scores.mean(axis = 1)
#     plt.figure(figsize = (16,10))
#     plt.plot(train_sizes, train_scores_mean, label = 'Training error')
#     plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

#     plt.ylabel('MSE', fontsize = 14)
#     plt.xlabel('Training set size', fontsize = 14)
#     title = 'Learning curves. MSE. ' + stringg
#     plt.title(title, fontsize = 18, y = 1.03)
#     plt.legend()
#     # plt.ylim(0,40)
#     # plt.savefig('fig_LinPol/MSELearnCurv' + str(stringg) +'.png')
#     train_sizes, train_scores, validation_scores = learning_curve(
#     estimator, data, target, train_sizes = train_sizes,
#     cv = cv, scoring = 'neg_mean_absolute_percentage_error')
#     train_scores_mean = -train_scores.mean(axis = 1)
#     validation_scores_mean = -validation_scores.mean(axis = 1)
#     plt.figure(figsize = (16,10))
#     plt.plot(train_sizes, train_scores_mean, label = 'Training error')
#     plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
#     plt.ylabel('MAPE', fontsize = 14)
#     plt.xlabel('Training set size', fontsize = 14)
#     title = 'Learning curves. MAPE. ' + stringg
#     plt.title(title, fontsize = 18, y = 1.03)
#     plt.legend()
    # plt.savefig('fig_LinPol/MAPELearnCurv' + str(stringg) +'.png')
    

# learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 30, 'no hour')

# for cv_par in range(2,30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' no hour ')

# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "sc_pol, degree = " + str(Degree),  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

                                                                # Model Scaled with hour and weekday

data["hour"] = df_new['datetime'].dt.hour



All_Test_size = 2304        

MeanCo_hour = np.array([np.NaN] * 24)
ar_hour = range(24)
for i in range(24):
    MeanCo_hour[i] = data[data['hour'] == i]['CO(GT)'].mean()

print(MeanCo_hour)
data["hour"] = df_new['datetime'].dt.hour
data["CO_Q1"] = MeanCo_hour[data["hour"]]
# data["weekday"] = df_new['datetime'].dt.weekday
y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)



X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=All_Test_size)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)



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

COmean_train_scaled = X_train_scaled.dropna()['CO_Q1']
COmean_test_scaled = X_test_scaled.dropna()['CO_Q1']
X_train_scaled = X_train_scaled.dropna().drop(['CO_Q1'], axis=1)
X_test_scaled = X_test_scaled.dropna().drop(['CO_Q1'], axis=1)

poly = PolynomialFeatures(degree=Degree)

X_train_poly_scaled = pd.DataFrame(poly.fit_transform(X_train_scaled))
X_test_poly_scaled = pd.DataFrame(poly.fit_transform(X_test_scaled))
list_poly_scaled =list(poly.get_feature_names_out())
X_train_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
X_test_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
X_train_poly_scaled = X_train_poly_scaled.dropna().drop(list_delete, axis=1)
X_test_poly_scaled  = X_test_poly_scaled.dropna().drop(list_delete, axis=1)

# X_train_poly_scaled['hour'] = hour_train_scaled
# X_test_poly_scaled['hour'] = hour_test_scaled

X_train_poly_scaled['CO_Q1'] = COmean_train_scaled 
X_test_poly_scaled['CO_Q1'] = COmean_test_scaled







List_Train = range(2, All_Test_size//24, 3)
# print(List_Train)


MSE_Train = np.array([np.NaN]*len(List_Train))
MSE_Test = np.array([np.NaN]*len(List_Train))

MAPE_Train = np.array([np.NaN]*len(List_Train))
MAPE_Test = np.array([np.NaN]*len(List_Train))

MDAPE_Train = np.array([np.NaN]*len(List_Train))
MDAPE_Test = np.array([np.NaN]*len(List_Train))


GRE_Train = np.array([np.NaN]*len(List_Train))
GRE_Test = np.array([np.NaN]*len(List_Train))





for count, temp_size in enumerate(List_Train):
    # print(count,temp_size/24, temp_size)
    x_aa, X_Train_temp, x_yy, y_train_temp = timeseries_train_test_split(X_train_poly_scaled, y_train, test_size = All_Test_size - temp_size*24)

    lr = LinearRegression()
    lr.fit(X_Train_temp, y_train_temp)

    prediction_test = lr.predict(X_test_poly_scaled)
    # print(prediction_test)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test, prediction_test)
    MSE_Test[count] = mean_s_error(y_test, prediction_test)
    GRE_Test[count] = gate_rate_error(y_test, prediction_test)   
    MDAPE_Test[count] = median_absolute_percentage_error(y_test, prediction_test)

    prediction_train = lr.predict(X_Train_temp)
    MAPE_Train[count] = mean_absolute_percentage_error(y_train_temp, prediction_train)
    MSE_Train[count] = mean_s_error(y_train_temp, prediction_train)
    GRE_Train[count] = gate_rate_error(y_train_temp, prediction_train)   
    MDAPE_Train[count] = median_absolute_percentage_error(y_train_temp, prediction_train)
  


MSE_Train_non = np.copy(MSE_Train)
GRE_Train_non = np.copy(GRE_Train)
MAPE_Train_non = np.copy(MAPE_Train)

MDAPE_Train_non = np.copy(MDAPE_Train)



MSE_Test_non = np.copy(MSE_Test)
GRE_Test_non = np.copy(GRE_Test)
MAPE_Test_non = np.copy(MAPE_Test)

MDAPE_Test_non = np.copy(MDAPE_Test)





plt.figure(figsize = (16,10))
plt.plot(List_Train, MSE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MSE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MSE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, MAPE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MAPE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MAPE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, GRE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, GRE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("GRE, Learning curve")
plt.legend(loc="best")



# tscv = TimeSeriesSplit(n_splits=10)
from sklearn.model_selection import KFold
tscv = KFold(n_splits = 10)


from sklearn.linear_model import LassoCV, RidgeCV

for count, temp_size in enumerate(List_Train):
    # print(count,temp_size/24, temp_size)
    x_aa, X_Train_temp, x_yy, y_train_temp = timeseries_train_test_split(X_train_poly_scaled, y_train, test_size = All_Test_size - temp_size*24)

    print(temp_size, y_train_temp.min())
    lasso = LassoCV(alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000], cv=tscv)
    lasso.fit(X_Train_temp, y_train_temp)

    prediction_test = lasso.predict(X_test_poly_scaled)
    # print(prediction_test)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test, prediction_test)
    MSE_Test[count] = mean_s_error(y_test, prediction_test)
    GRE_Test[count] = gate_rate_error(y_test, prediction_test)   
    MDAPE_Test[count] = median_absolute_percentage_error(y_test, prediction_test)


    prediction_train = lasso.predict(X_Train_temp)
    MAPE_Train[count] = mean_absolute_percentage_error(y_train_temp, prediction_train)
    MSE_Train[count] = mean_s_error(y_train_temp, prediction_train)
    GRE_Train[count] = gate_rate_error(y_train_temp, prediction_train)   
    MDAPE_Train[count] = median_absolute_percentage_error(y_train_temp, prediction_train)
  


MSE_Train_L1 = np.copy(MSE_Train)
GRE_Train_L1 = np.copy(GRE_Train)
MAPE_Train_L1 = np.copy(MAPE_Train)
MDAPE_Train_L1 = np.copy(MDAPE_Train)



MSE_Test_L1 = np.copy(MSE_Test)
GRE_Test_L1 = np.copy(GRE_Test)
MAPE_Test_L1 = np.copy(MAPE_Test)
MDAPE_Test_L1 = np.copy(MDAPE_Test)






plt.figure(figsize = (16,10))
plt.plot(List_Train, MSE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MSE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MSE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, MAPE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MAPE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MAPE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, GRE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, GRE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("GRE, Learning curve")
plt.legend(loc="best")










figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MAPE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MAPE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("MAPE, Learning curve")
axis[1].legend(loc="best")



figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MDAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MDAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MDAPE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MDAPE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("MdAPE, Learning curve")
axis[1].legend(loc="best")





figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, GRE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, GRE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, GRE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, GRE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("GRE, Learning curve")
axis[1].legend(loc="best")


























botom_limit = 0.15
upper_limit = 12









for count, temp_size in enumerate(List_Train):
    # print(count,temp_size/24, temp_size)
    x_aa, X_Train_temp_temp, x_yy, y_train_temp_temp = timeseries_train_test_split(X_train_poly_scaled, y_train, test_size = All_Test_size - temp_size*24)


    ALL_Temp = pd.DataFrame(X_Train_temp_temp.copy())
    ALL_Temp['CO(GT)'] = y_train_temp_temp.to_numpy()

    
    ALL_Temp = ALL_Temp[ALL_Temp['CO(GT)'] > botom_limit]
    ALL_Temp = ALL_Temp[ALL_Temp['CO(GT)'] < upper_limit]



    y_train_temp = ALL_Temp.dropna()['CO(GT)']
    X_Train_temp = ALL_Temp.dropna().drop(['CO(GT)'], axis = 1)


    print(temp_size*24, X_Train_temp.shape[0], y_train_temp.min(), y_train_temp.max())
    lr = LinearRegression()
    lr.fit(X_Train_temp, y_train_temp)

    prediction_test = lr.predict(X_test_poly_scaled)
    # print(prediction_test)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test, prediction_test)
    MSE_Test[count] = mean_s_error(y_test, prediction_test)
    GRE_Test[count] = gate_rate_error(y_test, prediction_test)   
    MDAPE_Test[count] = median_absolute_percentage_error(y_test, prediction_test)

    prediction_train = lr.predict(X_Train_temp)
    MAPE_Train[count] = mean_absolute_percentage_error(y_train_temp, prediction_train)
    MSE_Train[count] = mean_s_error(y_train_temp, prediction_train)
    GRE_Train[count] = gate_rate_error(y_train_temp, prediction_train)   
    MDAPE_Train[count] = median_absolute_percentage_error(y_train_temp, prediction_train)
  


MSE_Train_non_drop = np.copy(MSE_Train)
GRE_Train_non_drop = np.copy(GRE_Train)
MAPE_Train_non_drop = np.copy(MAPE_Train)

MDAPE_Train_non_drop = np.copy(MDAPE_Train)



MSE_Test_non_drop = np.copy(MSE_Test)
GRE_Test_non_drop = np.copy(GRE_Test)
MAPE_Test_non_drop = np.copy(MAPE_Test)

MDAPE_Test_non_drop = np.copy(MDAPE_Test)

















plt.figure(figsize = (16,10))
plt.plot(List_Train, MSE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MSE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MSE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, MAPE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MAPE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MAPE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, GRE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, GRE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("GRE, Learning curve")
plt.legend(loc="best")



# tscv = TimeSeriesSplit(n_splits=10)
from sklearn.model_selection import KFold
tscv = KFold(n_splits = 10)


from sklearn.linear_model import LassoCV, RidgeCV

for count, temp_size in enumerate(List_Train):
    # print(count,temp_size/24, temp_size)
    x_aa, X_Train_temp_temp, x_yy, y_train_temp_temp = timeseries_train_test_split(X_train_poly_scaled, y_train, test_size = All_Test_size - temp_size*24)


    ALL_Temp = pd.DataFrame(X_Train_temp_temp.copy())
    ALL_Temp['CO(GT)'] = y_train_temp_temp.to_numpy()

    
    ALL_Temp = ALL_Temp[ALL_Temp['CO(GT)'] > botom_limit]
    ALL_Temp = ALL_Temp[ALL_Temp['CO(GT)'] < upper_limit]



    y_train_temp = ALL_Temp.dropna()['CO(GT)']
    X_Train_temp = ALL_Temp.dropna().drop(['CO(GT)'], axis = 1)


    print(temp_size*24, X_Train_temp.shape[0], y_train_temp.min(), y_train_temp.max())
    lasso = LassoCV(alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000], cv=tscv)
    lasso.fit(X_Train_temp, y_train_temp)

    prediction_test = lasso.predict(X_test_poly_scaled)
    # print(prediction_test)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test, prediction_test)
    MSE_Test[count] = mean_s_error(y_test, prediction_test)
    GRE_Test[count] = gate_rate_error(y_test, prediction_test)   
    MDAPE_Test[count] = median_absolute_percentage_error(y_test, prediction_test)


    prediction_train = lasso.predict(X_Train_temp)
    MAPE_Train[count] = mean_absolute_percentage_error(y_train_temp, prediction_train)
    MSE_Train[count] = mean_s_error(y_train_temp, prediction_train)
    GRE_Train[count] = gate_rate_error(y_train_temp, prediction_train)   
    MDAPE_Train[count] = median_absolute_percentage_error(y_train_temp, prediction_train)
  


MSE_Train_L1_drop = np.copy(MSE_Train)
GRE_Train_L1_drop = np.copy(GRE_Train)
MAPE_Train_L1_drop = np.copy(MAPE_Train)
MDAPE_Train_L1_drop = np.copy(MDAPE_Train)



MSE_Test_L1_drop = np.copy(MSE_Test)
GRE_Test_L1_drop = np.copy(GRE_Test)
MAPE_Test_L1_drop = np.copy(MAPE_Test)
MDAPE_Test_L1_drop = np.copy(MDAPE_Test)






plt.figure(figsize = (16,10))
plt.plot(List_Train, MSE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MSE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MSE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, MAPE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, MAPE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("MAPE, Learning curve")
plt.legend(loc="best")

plt.figure(figsize = (16,10))
plt.plot(List_Train, GRE_Test, label="Test", linewidth=2.0, color = "red")
plt.plot(List_Train, GRE_Train, label="Train", linewidth=2.0, color = "black")
plt.title("GRE, Learning curve")
plt.legend(loc="best")























































# for count, temp_size in enumerate(List_Train):
#     # print(count,temp_size/24, temp_size)
#     x_aa, X_Train_temp, x_yy, y_train_temp = timeseries_train_test_split(X_train_poly_scaled, y_train, test_size = All_Test_size - temp_size*24)

#     ridge = RidgeCV(alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000], cv=tscv)
#     ridge.fit(X_Train_temp, y_train_temp)

#     prediction_test = ridge.predict(X_test_poly_scaled)
#     # print(prediction_test)
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test, prediction_test)
#     MSE_Test[count] = mean_s_error(y_test, prediction_test)
#     GRE_Test[count] = gate_rate_error(y_test, prediction_test)   

#     prediction_train = ridge.predict(X_Train_temp)
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train_temp, prediction_train)
#     MSE_Train[count] = mean_s_error(y_train_temp, prediction_train)
#     GRE_Train[count] = gate_rate_error(y_train_temp, prediction_train)   




# MSE_Train_L2 = MSE_Train
# GRE_Train_L2 = GRE_Train
# MAPE_Train_L2 = MAPE_Train

# MSE_Test_L2 = MSE_Test
# GRE_Test_L2 = GRE_Test
# MAPE_Test_L2 = MAPE_Test






# plt.figure(figsize = (16,10))
# plt.plot(List_Train, MSE_Test, label="Test", linewidth=2.0, color = "red")
# plt.plot(List_Train, MSE_Train, label="Train", linewidth=2.0, color = "black")
# plt.title("MSE, Learning curve")
# plt.legend(loc="best")

# plt.figure(figsize = (16,10))
# plt.plot(List_Train, MAPE_Test, label="Test", linewidth=2.0, color = "red")
# plt.plot(List_Train, MAPE_Train, label="Train", linewidth=2.0, color = "black")
# plt.title("MAPE, Learning curve")
# plt.legend(loc="best")

# plt.figure(figsize = (16,10))
# plt.plot(List_Train, GRE_Test, label="Test", linewidth=2.0, color = "red")
# plt.plot(List_Train, GRE_Train, label="Train", linewidth=2.0, color = "black")
# plt.title("GRE, Learning curve")
# plt.legend(loc="best")






figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MAPE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MAPE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("MAPE, Learning curve")
axis[1].legend(loc="best")



figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MDAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MDAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MDAPE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MDAPE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("MdAPE, Learning curve")
axis[1].legend(loc="best")





figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, GRE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, GRE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, GRE_Train_non, label="Train Non Reg", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, GRE_Train_L1, label="Train L1 Reg", linewidth=2.0, color = "darkorange")

axis[1].set_title("GRE, Learning curve")
axis[1].legend(loc="best")






figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")



axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MAPE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MAPE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")

axis[1].set_title("MAPE, Learning curve")
axis[1].legend(loc="best")



figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, MDAPE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, MDAPE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, MDAPE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, MDAPE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")

axis[1].set_title("MdAPE, Learning curve")
axis[1].legend(loc="best")





figure, axis = plt.subplots(1,2)

axis[0].plot(List_Train, MSE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[0].plot(List_Train, MSE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[0].plot(List_Train, MSE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[0].plot(List_Train, MSE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")

axis[0].set_title("MSE, Learning curve")
axis[0].legend(loc="best")


axis[1].plot(List_Train, GRE_Test_non, label="Test Non Reg", linewidth=2.0, color = "green")
axis[1].plot(List_Train, GRE_Test_L1, label="Test L1 Reg", linewidth=2.0, color = "red")
axis[1].plot(List_Train, GRE_Test_non_drop, label="Test Non Reg drop", linewidth=2.0, color = "lime")
axis[1].plot(List_Train, GRE_Test_L1_drop, label="Test L1 Reg drop", linewidth=2.0, color = "darkorange")

axis[1].set_title("GRE, Learning curve")
axis[1].legend(loc="best")



































tscv = TimeSeriesSplit(n_splits = 10)



X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=2000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)



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

COmean_train_scaled = X_train_scaled.dropna()['CO_Q1']
COmean_test_scaled = X_test_scaled.dropna()['CO_Q1']
X_train_scaled = X_train_scaled.dropna().drop(['CO_Q1'], axis=1)
X_test_scaled = X_test_scaled.dropna().drop(['CO_Q1'], axis=1)

poly = PolynomialFeatures(degree=Degree)

X_train_poly_scaled = pd.DataFrame(poly.fit_transform(X_train_scaled))
X_test_poly_scaled = pd.DataFrame(poly.fit_transform(X_test_scaled))
list_poly_scaled =list(poly.get_feature_names_out())
X_train_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
X_test_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
X_train_poly_scaled = X_train_poly_scaled.dropna().drop(list_delete, axis=1)
X_test_poly_scaled  = X_test_poly_scaled.dropna().drop(list_delete, axis=1)

# X_train_poly_scaled['hour'] = hour_train_scaled
# X_test_poly_scaled['hour'] = hour_test_scaled

X_train_poly_scaled['CO_Q1'] = COmean_train_scaled 
X_test_poly_scaled['CO_Q1'] = COmean_test_scaled


print(X_train_poly_scaled)
print(X_test_poly_scaled)


lr = LinearRegression()
lr.fit(X_train_poly_scaled, y_train)


plotModelResults(lr, X_train=X_train_poly_scaled, X_test=X_test_poly_scaled, string = "non Reg",  plot_intervals=True,\
                 plot_diff = True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
plotCoefficients(lr, X_train=X_train_poly_scaled)



lasso = LassoCV(alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000], cv=tscv)
lasso.fit(X_train_poly_scaled, y_train)


plotModelResults(lasso, X_train=X_train_poly_scaled, X_test=X_test_poly_scaled, string = "lasso",  plot_intervals=True,\
                 plot_diff = True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
plotCoefficients(lasso, X_train=X_train_poly_scaled)
































plt.show()





















# plotModelResults(lr, X_train=X_train_poly_scaled, X_test=X_test_poly_scaled, string = "sc_pol, degree = " + str(Degree), \
#                  plot_intervals=True, y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)
# plotCoefficients(lr, X_train=X_train_poly_scaled)





















# plt.figure(figsize = (16,10))
# learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 30, 'hour')




































# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

                                                                # Pol Scaled with hour and weekday
        
                                                            # LASSO, RIDGE

# from sklearn.linear_model import LassoCV, RidgeCV
# from sklearn.linear_model import Lasso, Ridge
#                                                      # Pol Ridge Scaled with hour and weekday

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(RidgeCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')


#                                                      # Pol LASSO Scaled with hour and weekday

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(LassoCV(cv=tscv), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')


                                                                # BOOSTING

# from xgboost import XGBRegressor

#                                                   # Pol XGBR Scaled with hour and weekday

# # xgb = XGBRegressor(verbosity=0)

# # # xgb.fit(X_train_scaled_poly, y_train)
# # # plotModelResults(xgb,X_train = X_train_scaled_poly,X_test=X_test_scaled_poly, plot_intervals=True, string ="sc_pol XGBR with h,d", y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test)

# for cv_par in range(2, 30, 10):
#     plt.figure(figsize = (16,10))
#     learning_curves(XGBRegressor(verbosity=0), X_all_scaled_poly, y_all,List_Train_Size, cv_par, ' +hour ')

# plt.plot(X_all_scaled_poly['hour'], y_all, label="C(hour)", marker = 'o',markersize=3, linestyle = 'None', color = "green")
    
# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'CO(GT)', data = data)
    
# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'T', data = data)

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'PT08.S1(CO)', data = data)

# plt.figure(figsize = (16,10))
# sns.boxplot(x = 'hour', y = 'PT08.S2(NMHC)', data = data)

# # sns.boxplot(x = 'T', y = 'CO(GT)', data = data)
    
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

# sns.jointplot(x = 'NO2op1', y = 'NO2op2', data = df_1, kind = 'scatter');

# plt.figure(figsize = (16,10))
# sns.scatterplot(data=data, x='CO(GT)', y='PT08.S1(CO)', hue = 'hour')

# for i in range(24):
#     # plt.figure(figsize = (16,10))
#     sns.jointplot(data=data[data['hour'] == i], x='CO(GT)', y='PT08.S1(CO)', kind = 'reg')

plt.show()

