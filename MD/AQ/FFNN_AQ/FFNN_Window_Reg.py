# Анализ количества нейронов в скрытом слое 
# для tanh, sigmoid в диапазоне от 1 до 22 нейронов
# Сравнение моделей без регуляризации и с L1,L2 
# регуляризацией. Параметр регуляризации - базовый 0.01
# Регуляризация почти всюду лучше.

import matplotlib.pyplot as plt  # plots
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
import seaborn as sns  # more plots
sns.set()

import warnings  
warnings.filterwarnings('ignore')

from itertools import product  # some useful functions

from tensorflow.keras import regularizers


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
from tensorflow import keras


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




feat_CO = ['PT08.S1(CO)','T', 'PT08.S2(NMHC)', 'CO(GT)']

# feat_CO = ['PT08.S1(CO)','T', 'RH', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)','T', 'PT08.S2(NMHC)','PT08.S3(NOx)','CO(GT)']
# feat_CO = ['T','CO(GT)']


Degree = 1
list_delete = []
# list_delete = ['T^2','R_NM^2']
# list_delete = ['T^2','PT08.S2(NMHC)^2']


T_size = 2000

List_Train_Size = np.arange(0.01,1.0,0.01)





feat_target = feat_CO[-1]
l_feat = len(feat_CO) - 1

for feat in feat_CO:
    df_new = df_new[df_new[feat] > -100]



# df_new = df_new[df_new['CO(GT)'] > 0.4]
# df_new = df_new[df_new['CO(GT)'] < 5.0]

print(df_new['CO(GT)'].min())
print(df_new['CO(GT)'].max())

data = pd.DataFrame(df_new[['datetime'] + feat_CO].copy())
print(data.tail(7))


data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)



y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mean_s_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

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



X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
time_train = pd.DataFrame()
time_test = pd.DataFrame()
temp_str = " "
def plotModelResults(
    model, X_train=X_train, X_test=X_test,string = temp_str, plot_intervals=False,\
    plot_diff = False, plot_anomalies=False, time_test = time_test,\
    time_train = time_train, y_test = y_test, y_train = y_train
):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test).flatten()

    # print(str(model))
    # print("prediction:")
    # print(prediction.flatten())
    # print("y_test:") 
    # print(y_test)
    # print(prediction.flatten()-y_test)
    # print(str(model))


    plt.figure(figsize=(15, 7))
    # plt.plot(time_test, prediction, label="prediction", marker = 'o',markersize=3, linestyle = 'None', color = "green")
    # plt.plot(time_test, y_test.values, label="actual",  marker = 'o',markersize=3, linestyle = 'None', color = "black")

    plt.plot(time_test, prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(time_test, y_test.values, label="actual", linewidth=2.0)
    # plt.savefig('fig_LinPol/Pred_True_' + str(string) +'.png')



    # if plot_intervals:
    #     cv = cross_val_score(
    #         model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
    #     )
    #     mae = cv.mean() * (-1)
    #     deviation = cv.std()

    #     scale = 1.96
    #     lower = prediction - (mae + scale * deviation)
    #     upper = prediction + (mae + scale * deviation)

        # plt.plot(time_test, lower, "r--", label="upper bond / lower bond", alpha=0.5, )
        # plt.plot(time_test, upper, "r--", alpha=0.5)

        # if plot_anomalies:
        #     anomalies = np.array([np.NaN] * len(y_test))
        #     anomalies[y_test < lower] = y_test[y_test < lower]
        #     anomalies[y_test > upper] = y_test[y_test > upper]
        #     plt.plot(time_test, anomalies, "o", markersize=10, linestyle = 'None',label="Anomalies")



    error_mape = mean_absolute_percentage_error(y_test.values,prediction)
    array_diff = abs(y_test.values - prediction)
    array_mse  = (y_test.values - prediction)**2
    array_mape = (array_diff/y_test.values) * 100
    array_gre  = arr_gate_rate_error(y_test.values,prediction)
    error_mse = mean_s_error(y_test.values,prediction)
    error_gre = gate_rate_error(y_test.values,prediction)

    plt.title("MAPE: {0:.2f}% ".format(error_mape) + "MSE: {0:.2f} ".format(error_mse) + "GRE(25%): {0:.2f}% ".format(error_gre) + str(string))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)

    if plot_diff:
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
        # temp['CO_hour'] = X_test['CO_hour']
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
        
                                                                        # Plot Prediction (True)
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

                                                                            # Plot MSE ERROR

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

                                                                            # Plot MAPE ERROR
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

                                                                            # Plot GRE ERROR

        # plt.figure(figsize=(15, 7))
        # plt.plot(temp['hour'], array_mape, label="MSE",  marker = 'o',markersize=3, linestyle = 'None', color = "black")
        # plt.title("MAPE(hour)" + str(string))
        # plt.legend(loc="best")
        # plt.tight_layout()
        # plt.grid(True)

        # plt.figure(figsize = (16,10))
        # sns.boxplot(x = 'CO_hour', y = 'MAPE', data = temp)
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

        # plt.figure(figsize = (16,10))
        # for mon in range(1,13):
        #     plt.plot(mon, temp[temp['Month'] == mon]['GRE'].mean())

def plotCoefficients(model, X_train = X_train):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed");


from sklearn.model_selection import train_test_split
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,cross_val_score)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import learning_curve

def learning_curves(estimator, data, target, train_sizes, cv, stringg):
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, data, target, train_sizes = train_sizes,
    cv = cv, scoring = 'neg_mean_squared_error')
    # 'neg_mean_squared_error''neg_mean_absolute_percentage_error'
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.figure(figsize = (16,10))
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + str(stringg)+' model, cv_par = ' + str(cv)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    # plt.ylim(0,40)
    # plt.savefig('fig_LinPol/MSELearnCurv' + str(stringg) +'.png')

    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, data, target, train_sizes = train_sizes,
    cv = cv, scoring = 'neg_mean_absolute_percentage_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.figure(figsize = (16,10))
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MAPE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + str(stringg)+' model, cv_par = ' + str(cv)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    # plt.savefig('fig_LinPol/MAPELearnCurv' + str(stringg) +'.png')
    

def PolyScaledData(data_df = df_new, target_name = feat_target, pol_deg = Degree, TimeSplit = True, Test_size = T_size, add_hour = False, add_CO_hour = False, list_drop = list_delete, add_window = False, size_window = 24):
    if add_hour:
        data["hour"] = data_df['datetime'].dt.hour
    if add_CO_hour:
        data["hour"] = data_df['datetime'].dt.hour
        CO_hour = np.array([np.NaN] * 24)
        ar_hour = range(24)
        for i in range(24):
            CO_hour[i] = data[data['hour'] == i]['CO(GT)'].mean()
        data["CO_hour"] = CO_hour[data["hour"]]
    # print(CO_hour)
    # print(data.head(30))
    y = data.dropna()[target_name]
    X = data.dropna().drop([target_name], axis=1)

    if TimeSplit:
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=Test_size)
    else:
        print("aaaa")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Test_size)
    time_train = X_train.dropna()['datetime']
    time_test = X_test.dropna()['datetime']
    X_train = X_train.dropna().drop(['datetime'], axis=1)
    X_test = X_test.dropna().drop(['datetime'], axis=1)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    list_X = X_test.columns.values.tolist()
    X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
    X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)
    if add_hour:
        hour_train_scaled = X_train_scaled.dropna()['hour']
        hour_test_scaled = X_test_scaled.dropna()['hour']
        X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
        X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)
    if add_CO_hour:
        hour_train_scaled = X_train_scaled.dropna()['hour']
        hour_test_scaled = X_test_scaled.dropna()['hour']
        X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
        X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)
        CO_hour_train_scaled = X_train_scaled.dropna()['CO_hour']
        CO_hour_test_scaled = X_test_scaled.dropna()['CO_hour']
        X_train_scaled = X_train_scaled.dropna().drop(['CO_hour'], axis=1)
        X_test_scaled = X_test_scaled.dropna().drop(['CO_hour'], axis=1)
    poly = PolynomialFeatures(degree=pol_deg)
    X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
    X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
    list_scaled_poly =list(poly.get_feature_names_out())
    X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
    X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
    X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_drop, axis=1)
    X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_drop, axis=1)
    if add_hour:
        X_train_scaled_poly['hour'] = hour_train_scaled
        X_test_scaled_poly['hour'] = hour_test_scaled
    if add_CO_hour:
        X_train_scaled_poly['CO_hour'] = CO_hour_train_scaled 
        X_test_scaled_poly['CO_hour'] = CO_hour_test_scaled
    if add_window:                
        X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True)
        y_all = y_train.append(y_test, ignore_index=True) 
        
        Window = np.array([np.NaN] * X_all_scaled_poly.shape[0])
            
                                            # R_CO

        for j in range(X_all_scaled_poly.shape[0]):
            if j > (size_window - 1):
                Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j-(size_window - 1), j+1))/size_window
            else:
                Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1))/(j+1.0)    
        X_all_scaled_poly['WR_CO'] = Window
        X_all_scaled_poly['DR_CO'] = X_all_scaled_poly['R_CO'] - X_all_scaled_poly['WR_CO']
        X_all_scaled_poly = X_all_scaled_poly.dropna().drop(['R_CO'], axis=1)

                                            # R_NM

        for j in range(X_all_scaled_poly.shape[0]):
            if j > (size_window - 1):
                Window[j] = sum(X_all_scaled_poly['R_NM'][alpha] for alpha in range(j-(size_window - 1), j+1))/size_window
            else:
                Window[j] = sum(X_all_scaled_poly['R_NM'][alpha] for alpha in range(j+1))/(j+1.0)    
        X_all_scaled_poly['WR_NM'] = Window
        X_all_scaled_poly['DR_NM'] = X_all_scaled_poly['R_NM'] - X_all_scaled_poly['WR_NM']
        X_all_scaled_poly = X_all_scaled_poly.dropna().drop(['R_NM'], axis=1)

                                            # T

        for j in range(X_all_scaled_poly.shape[0]):
            if j > (size_window - 1):
                Window[j] = sum(X_all_scaled_poly['T'][alpha] for alpha in range(j-(size_window - 1), j+1))/size_window
            else:
                Window[j] = sum(X_all_scaled_poly['T'][alpha] for alpha in range(j+1))/(j+1.0)    
        X_all_scaled_poly['W_T'] = Window
        X_all_scaled_poly['D_T'] = X_all_scaled_poly['T'] - X_all_scaled_poly['W_T']
        X_all_scaled_poly = X_all_scaled_poly.dropna().drop(['T'], axis=1)


        # for j in range(X_all_scaled_poly.shape[0]):
        #     if j > (size_window - 1):
        #         Window[j] = sum(X_all_scaled_poly['RH'][alpha] for alpha in range(j-(size_window - 1), j+1))/size_window
        #     else:
        #         Window[j] = sum(X_all_scaled_poly['RH'][alpha] for alpha in range(j+1))/(j+1.0)    
        # X_all_scaled_poly['W_RH'] = Window
        # X_all_scaled_poly['D_RH'] = X_all_scaled_poly['RH'] - X_all_scaled_poly['W_RH']
        # X_all_scaled_poly = X_all_scaled_poly.dropna().drop(['RH'], axis=1)





        X_train_scaled_poly, X_test_scaled_poly, y_train, y_test = timeseries_train_test_split(X_all_scaled_poly, y_all, test_size=Test_size)

    # X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
    # X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)

    return X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test



from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression

from sklearn import preprocessing

from sklearn.model_selection import RepeatedKFold
kfold = RepeatedKFold(n_splits=5, n_repeats=100)
from ann_visualizer.visualize import ann_viz

# Type_error = 'mape'

List_Error = ['mean_absolute_percentage_error', 'mean_squared_error']
Type_error = 'mean_squared_error'

# ['softmax', 'softplus', 'softsign']
# Type_func = List_Function[0]
# Type_optimizer='adam'   
Type_optimizer='RMSprop'
Number_epochs = 100

x = np.linspace(-1, 1)
plt.figure(figsize = (16,10))
plt.plot(x, keras.activations.linear(x),label="linear", color = 'red') 
plt.plot(x, keras.activations.elu(x),label="elu", color = 'chocolate') 
plt.plot(x, keras.activations.relu(x),label="relu", color = 'orange') 
plt.plot(x, keras.activations.selu(x),label="selu", color = 'gold') 
plt.plot(x, keras.activations.sigmoid(x),label="sigmoid", color = 'green') 
plt.plot(x, keras.activations.tanh(x),label="tanh", color = 'dodgerblue') 
plt.plot(x, keras.activations.exponential(x),label="exponential", color = 'indigo')  

plt.title("Activations Function")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)



                                                                        # ML Model


# X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_hour = True, add_window = False)
# # X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_CO_hour = True)


# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)

# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "ML Pol" + str(Degree),  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# plotCoefficients(lr, X_train=X_train_scaled_poly)


#                                                                         # NN Linear Model Adam

# network = models.Sequential()
# l_feat = X_train_scaled_poly.shape[1]
# network.add(layers.Dense(units = 1, activation = 'linear', input_shape=(l_feat,)))
# network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.001), metrics=['mean_squared_error']) 
# history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 500, validation_split = 0)

def plot_loss(history, string_l = 'Loss', draw_val = False):
    plt.figure(figsize = (16,10))
    plt.plot(history.history['loss'], label='loss')
    if draw_val:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(string_l)
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')   
    plt.legend()
    plt.grid(True)

# plot_loss(history, string_l = "NN(ML) Adam 0.001")
# plotModelResults(network, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "NN(ML) Adam 0.001,  No Window",  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# keras.backend.clear_session()












X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_CO_hour = True, add_window = True)
# X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_hour = True, add_window = True)
# X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_window = True)
# X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_CO_hour = True)

X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
y_all = y_train.append(y_test, ignore_index=True)
time_all = time_train.append(time_test, ignore_index = True)




network = models.Sequential()
l_feat = X_train_scaled_poly.shape[1]
network.add(layers.Dense(units = 1, activation = 'linear', input_shape=(l_feat,)))
network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.0001), metrics=['mean_squared_error']) 
history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 500, validation_split = 0)

def plot_loss(history, string_l = 'Loss', draw_val = False):
    plt.figure(figsize = (16,10))
    plt.plot(history.history['loss'], label='loss')
    if draw_val:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(string_l)
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')   
    plt.legend()
    plt.grid(True)

# plot_loss(history, string_l = "NN(ML) Adam 0.001")
plotModelResults(network, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "NN(ML) Adam 0.001,  Window = 1 day",  plot_intervals=True,\
                 y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
keras.backend.clear_session()


                                                                # MAE


print(X_train_scaled_poly)

List_reg = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5] 

List_MSE_Train_Min = np.array([np.NaN] * len(List_reg)) 
List_MSE_Train_Max = np.array([np.NaN] * len(List_reg)) 
List_MSE_Train_Mean = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Train_Min = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Train_Max = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Train_Mean = np.array([np.NaN] * len(List_reg)) 


List_MSE_Test_Min = np.array([np.NaN] * len(List_reg)) 
List_MSE_Test_Max = np.array([np.NaN] * len(List_reg)) 
List_MSE_Test_Mean = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Test_Min = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Test_Max = np.array([np.NaN] * len(List_reg)) 
List_MAPE_Test_Mean = np.array([np.NaN] * len(List_reg)) 


Number_measure = 10
MSE_Train = np.array([np.NaN] * Number_measure)
MAPE_Train = np.array([np.NaN] * Number_measure)
MSE_Test = np.array([np.NaN] * Number_measure)
MAPE_Test = np.array([np.NaN] * Number_measure)






# activ_func = 'tanh'
# loss_func = 'mean_absolute_error'

# for count, reg in enumerate(List_reg):
#     for iteration in range(Number_measure):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]
#         # print(iteration, l_feat)
#         network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#         network.add(layers.Dense(units=1, activation = 'linear'))
#         network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
#         if iteration == Number_measure - 1:
#             plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
#         MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     List_MSE_Train_Min[count] = MSE_Train.min() 
#     List_MSE_Train_Max[count] = MSE_Train.max() 
#     List_MSE_Train_Mean[count] = MSE_Train.mean() 
#     List_MAPE_Train_Min[count] = MAPE_Train.min() 
#     List_MAPE_Train_Max[count] = MAPE_Train.max() 
#     List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

#     List_MSE_Test_Min[count] = MSE_Test.min() 
#     List_MSE_Test_Max[count] = MSE_Test.max() 
#     List_MSE_Test_Mean[count] = MSE_Test.mean() 
#     List_MAPE_Test_Min[count] = MAPE_Test.min() 
#     List_MAPE_Test_Max[count] = MAPE_Test.max()  
#     List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
#     print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_reg, List_MSE_Train_Min, label="Train_min", linewidth=1.0, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Max, label="Train_max", linewidth=1.0, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")
# axis[0].plot(List_reg, List_MSE_Test_Min, label="Test_min", linewidth=1.0, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Max, label="Test_max", linewidth=1.0, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_mean", linewidth=2.0, color = "green")
# axis[0].set_title("MSE, Hidden_Tanh 10, non_regul")
# axis[0].legend(loc="best")



# axis[1].plot(List_reg, List_MAPE_Train_Min, label="Train_min", linewidth=1.0, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Max, label="Train_max", linewidth=1.0, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")
# axis[1].plot(List_reg, List_MAPE_Test_Min, label="Test_min", linewidth=1.0, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Max, label="Test_max", linewidth=1.0, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_mean", linewidth=2.0, color = "green")
# axis[1].set_title("MAPE, Hidden_Tanh 10, non_regul")
# axis[1].legend(loc="best")

 
# List_MSE_non_Min = np.copy(List_MSE_Test_Min)
# List_MSE_non_Max = np.copy(List_MSE_Test_Max)
# List_MSE_non_Mean = np.copy(List_MSE_Test_Mean) 
# List_MAPE_non_Min = np.copy(List_MAPE_Test_Min) 
# List_MAPE_non_Max = np.copy(List_MAPE_Test_Max)  
# List_MAPE_non_Mean = np.copy(List_MAPE_Test_Mean)  





# for count, reg in enumerate(List_reg):
#     for iteration in range(Number_measure):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]
#         # print(iteration, l_feat)
#         network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L1(10**reg)))
#         network.add(layers.Dense(units=1, activation = 'linear'))
#         network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
#         if iteration == Number_measure - 1:
#             plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
#         MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     List_MSE_Train_Min[count] = MSE_Train.min() 
#     List_MSE_Train_Max[count] = MSE_Train.max() 
#     List_MSE_Train_Mean[count] = MSE_Train.mean() 
#     List_MAPE_Train_Min[count] = MAPE_Train.min() 
#     List_MAPE_Train_Max[count] = MAPE_Train.max() 
#     List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

#     List_MSE_Test_Min[count] = MSE_Test.min() 
#     List_MSE_Test_Max[count] = MSE_Test.max() 
#     List_MSE_Test_Mean[count] = MSE_Test.mean() 
#     List_MAPE_Test_Min[count] = MAPE_Test.min() 
#     List_MAPE_Test_Max[count] = MAPE_Test.max()  
#     List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
#     print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

# axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[0].set_title("MSE, Hidden_Tanh 10, L1_regul")
# axis[0].legend(loc="best")



# axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")


# axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[1].set_title("MAPE, Hidden_Tanh 10, L1_regul")
# axis[1].legend(loc="best")


# List_MSE_L1_Min = np.copy(List_MSE_Test_Min)
# List_MSE_L1_Max = np.copy(List_MSE_Test_Max)
# List_MSE_L1_Mean = np.copy(List_MSE_Test_Mean) 
# List_MAPE_L1_Min = np.copy(List_MAPE_Test_Min) 
# List_MAPE_L1_Max = np.copy(List_MAPE_Test_Max)  
# List_MAPE_L1_Mean = np.copy(List_MAPE_Test_Mean)  






# for count, reg in enumerate(List_reg):
#     for iteration in range(Number_measure):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]
#         # print(iteration, l_feat)
#         network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L2(10**reg)))
#         network.add(layers.Dense(units=1, activation = 'linear'))
#         network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
#         if iteration == Number_measure - 1:
#             plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
#         MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     List_MSE_Train_Min[count] = MSE_Train.min() 
#     List_MSE_Train_Max[count] = MSE_Train.max() 
#     List_MSE_Train_Mean[count] = MSE_Train.mean() 
#     List_MAPE_Train_Min[count] = MAPE_Train.min() 
#     List_MAPE_Train_Max[count] = MAPE_Train.max() 
#     List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

#     List_MSE_Test_Min[count] = MSE_Test.min() 
#     List_MSE_Test_Max[count] = MSE_Test.max() 
#     List_MSE_Test_Mean[count] = MSE_Test.mean() 
#     List_MAPE_Test_Min[count] = MAPE_Test.min() 
#     List_MAPE_Test_Max[count] = MAPE_Test.max()  
#     List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
#     print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

# axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")


# axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[0].set_title("MSE, Hidden_Tanh 10, L2_regul")
# axis[0].legend(loc="best")



# axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

# axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[1].set_title("MAPE, Hidden_Tanh 10, L2_regul")
# axis[1].legend(loc="best")


# List_MSE_L2_Min = np.copy(List_MSE_Test_Min)
# List_MSE_L2_Max = np.copy(List_MSE_Test_Max)
# List_MSE_L2_Mean = np.copy(List_MSE_Test_Mean) 
# List_MAPE_L2_Min = np.copy(List_MAPE_Test_Min) 
# List_MAPE_L2_Max = np.copy(List_MAPE_Test_Max)  
# List_MAPE_L2_Mean = np.copy(List_MAPE_Test_Mean)  








# for count, reg in enumerate(List_reg):
#     for iteration in range(Number_measure):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]
#         # print(iteration, l_feat)
#         network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L1L2(l1 = 10**reg, l2 = 10**reg)))
#         network.add(layers.Dense(units=1, activation = 'linear'))
#         network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
#         if iteration == Number_measure - 1:
#             plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
#         MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     List_MSE_Train_Min[count] = MSE_Train.min() 
#     List_MSE_Train_Max[count] = MSE_Train.max() 
#     List_MSE_Train_Mean[count] = MSE_Train.mean() 
#     List_MAPE_Train_Min[count] = MAPE_Train.min() 
#     List_MAPE_Train_Max[count] = MAPE_Train.max() 
#     List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

#     List_MSE_Test_Min[count] = MSE_Test.min() 
#     List_MSE_Test_Max[count] = MSE_Test.max() 
#     List_MSE_Test_Mean[count] = MSE_Test.mean() 
#     List_MAPE_Test_Min[count] = MAPE_Test.min() 
#     List_MAPE_Test_Max[count] = MAPE_Test.max()  
#     List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
#     print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

# axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")


# axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[0].set_title("MSE, Hidden_Tanh 10, L1L2_regul")
# axis[0].legend(loc="best")



# axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

# axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

# axis[1].set_title("MAPE, Hidden_Tanh 10, L1L2_regul")
# axis[1].legend(loc="best")


# List_MSE_L1L2_Min = np.copy(List_MSE_Test_Min)
# List_MSE_L1L2_Max = np.copy(List_MSE_Test_Max)
# List_MSE_L1L2_Mean = np.copy(List_MSE_Test_Mean) 
# List_MAPE_L1L2_Min = np.copy(List_MAPE_Test_Min) 
# List_MAPE_L1L2_Max = np.copy(List_MAPE_Test_Max)  
# List_MAPE_L1L2_Mean = np.copy(List_MAPE_Test_Mean)  




# figure, axis = plt.subplots(1,2)

# axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
# axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[0].plot(List_reg, List_MSE_L1_Min, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_L1_Max, linewidth=0.75, color = "red")
# axis[0].plot(List_reg, List_MSE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")

# axis[0].plot(List_reg, List_MSE_L2_Min, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_L2_Max, linewidth=0.75, color = "green")
# axis[0].plot(List_reg, List_MSE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")

# axis[0].plot(List_reg, List_MSE_L1L2_Min, linewidth=0.75, color = "blue")
# axis[0].plot(List_reg, List_MSE_L1L2_Max, linewidth=0.75, color = "blue")
# axis[0].plot(List_reg, List_MSE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")


# axis[0].set_title("MSE, Hidden_Tanh 10")
# axis[0].legend(loc="best")



# axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
# axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

# axis[1].plot(List_reg, List_MAPE_L1_Min, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_L1_Max, linewidth=0.75, color = "red")
# axis[1].plot(List_reg, List_MAPE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")

# axis[1].plot(List_reg, List_MAPE_L2_Min, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_L2_Max, linewidth=0.75, color = "green")
# axis[1].plot(List_reg, List_MAPE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")

# axis[1].plot(List_reg, List_MAPE_L1L2_Min, linewidth=0.75, color = "blue")
# axis[1].plot(List_reg, List_MAPE_L1L2_Max, linewidth=0.75, color = "blue")
# axis[1].plot(List_reg, List_MAPE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")

# axis[1].set_title("MAPE, Hidden_Tanh 10")
# axis[1].legend(loc="best")


# figure, axis = plt.subplots(1,2)

# axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")
# axis[0].plot(List_reg, List_MSE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")
# axis[0].plot(List_reg, List_MSE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")
# axis[0].plot(List_reg, List_MSE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")

# axis[0].set_title("MSE, Hidden_Tanh 10, Mean")
# axis[0].legend(loc="best")

# axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")
# axis[1].plot(List_reg, List_MAPE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")
# axis[1].plot(List_reg, List_MAPE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")
# axis[1].plot(List_reg, List_MAPE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")

# axis[1].set_title("MAPE, Hidden_Tanh 10, Mean")
# axis[1].legend(loc="best")



# figure, axis = plt.subplots(1,2)

# axis[0].plot(List_reg, List_MSE_non_Min, label="NonReg_Min", linewidth=2.0, color = "black")
# axis[0].plot(List_reg, List_MSE_L1_Min, label="L1Reg_Min", linewidth=2.0, color = "red")
# axis[0].plot(List_reg, List_MSE_L2_Min, label="L2Reg_Min", linewidth=2.0, color = "green")
# axis[0].plot(List_reg, List_MSE_L1L2_Min, label="L1L2Reg_Min", linewidth=2.0, color = "blue")

# axis[0].set_title("MSE, Hidden_Tanh 10, Min")
# axis[0].legend(loc="best")

# axis[1].plot(List_reg, List_MAPE_non_Min, label="NonReg_Min", linewidth=2.0, color = "black")
# axis[1].plot(List_reg, List_MAPE_L1_Min, label="L1Reg_Min", linewidth=2.0, color = "red")
# axis[1].plot(List_reg, List_MAPE_L2_Min, label="L2Reg_Min", linewidth=2.0, color = "green")
# axis[1].plot(List_reg, List_MAPE_L1L2_Min, label="L1L2Reg_Min", linewidth=2.0, color = "blue")

# axis[1].set_title("MAPE, Hidden_Tanh 10, Min")
# axis[1].legend(loc="best")






                 
                                                                        # SIGMOID
                 
 



activ_func = 'sigmoid'
loss_func = 'mean_absolute_error'



for count, reg in enumerate(List_reg):
    for iteration in range(Number_measure):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        # print(iteration, l_feat)
        network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
        if iteration == Number_measure - 1:
            plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
        MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        keras.backend.clear_session()
    List_MSE_Train_Min[count] = MSE_Train.min() 
    List_MSE_Train_Max[count] = MSE_Train.max() 
    List_MSE_Train_Mean[count] = MSE_Train.mean() 
    List_MAPE_Train_Min[count] = MAPE_Train.min() 
    List_MAPE_Train_Max[count] = MAPE_Train.max() 
    List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

    List_MSE_Test_Min[count] = MSE_Test.min() 
    List_MSE_Test_Max[count] = MSE_Test.max() 
    List_MSE_Test_Mean[count] = MSE_Test.mean() 
    List_MAPE_Test_Min[count] = MAPE_Test.min() 
    List_MAPE_Test_Max[count] = MAPE_Test.max()  
    List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
    print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


figure, axis = plt.subplots(1,2) 
axis[0].plot(List_reg, List_MSE_Train_Min, label="Train_min", linewidth=1.0, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Max, label="Train_max", linewidth=1.0, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")
axis[0].plot(List_reg, List_MSE_Test_Min, label="Test_min", linewidth=1.0, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Max, label="Test_max", linewidth=1.0, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_mean", linewidth=2.0, color = "green")
axis[0].set_title("MSE, Hidden_sigmoid 10, non_regul")
axis[0].legend(loc="best")



axis[1].plot(List_reg, List_MAPE_Train_Min, label="Train_min", linewidth=1.0, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Max, label="Train_max", linewidth=1.0, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")
axis[1].plot(List_reg, List_MAPE_Test_Min, label="Test_min", linewidth=1.0, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Max, label="Test_max", linewidth=1.0, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_mean", linewidth=2.0, color = "green")
axis[1].set_title("MAPE, Hidden_sigmoid 10, non_regul")
axis[1].legend(loc="best")

 
List_MSE_non_Min = np.copy(List_MSE_Test_Min)
List_MSE_non_Max = np.copy(List_MSE_Test_Max)
List_MSE_non_Mean = np.copy(List_MSE_Test_Mean) 
List_MAPE_non_Min = np.copy(List_MAPE_Test_Min) 
List_MAPE_non_Max = np.copy(List_MAPE_Test_Max)  
List_MAPE_non_Mean = np.copy(List_MAPE_Test_Mean)  





for count, reg in enumerate(List_reg):
    for iteration in range(Number_measure):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        # print(iteration, l_feat)
        network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L1(10**reg)))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
        if iteration == Number_measure - 1:
            plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
        MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        keras.backend.clear_session()
    List_MSE_Train_Min[count] = MSE_Train.min() 
    List_MSE_Train_Max[count] = MSE_Train.max() 
    List_MSE_Train_Mean[count] = MSE_Train.mean() 
    List_MAPE_Train_Min[count] = MAPE_Train.min() 
    List_MAPE_Train_Max[count] = MAPE_Train.max() 
    List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

    List_MSE_Test_Min[count] = MSE_Test.min() 
    List_MSE_Test_Max[count] = MSE_Test.max() 
    List_MSE_Test_Mean[count] = MSE_Test.mean() 
    List_MAPE_Test_Min[count] = MAPE_Test.min() 
    List_MAPE_Test_Max[count] = MAPE_Test.max()  
    List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
    print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


figure, axis = plt.subplots(1,2) 
axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[0].set_title("MSE, Hidden_sigmoid 10, L1_regul")
axis[0].legend(loc="best")



axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")


axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[1].set_title("MAPE, Hidden_sigmoid 10, L1_regul")
axis[1].legend(loc="best")


List_MSE_L1_Min = np.copy(List_MSE_Test_Min)
List_MSE_L1_Max = np.copy(List_MSE_Test_Max)
List_MSE_L1_Mean = np.copy(List_MSE_Test_Mean) 
List_MAPE_L1_Min = np.copy(List_MAPE_Test_Min) 
List_MAPE_L1_Max = np.copy(List_MAPE_Test_Max)  
List_MAPE_L1_Mean = np.copy(List_MAPE_Test_Mean)  






for count, reg in enumerate(List_reg):
    for iteration in range(Number_measure):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        # print(iteration, l_feat)
        network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L2(10**reg)))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
        if iteration == Number_measure - 1:
            plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
        MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        keras.backend.clear_session()
    List_MSE_Train_Min[count] = MSE_Train.min() 
    List_MSE_Train_Max[count] = MSE_Train.max() 
    List_MSE_Train_Mean[count] = MSE_Train.mean() 
    List_MAPE_Train_Min[count] = MAPE_Train.min() 
    List_MAPE_Train_Max[count] = MAPE_Train.max() 
    List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

    List_MSE_Test_Min[count] = MSE_Test.min() 
    List_MSE_Test_Max[count] = MSE_Test.max() 
    List_MSE_Test_Mean[count] = MSE_Test.mean() 
    List_MAPE_Test_Min[count] = MAPE_Test.min() 
    List_MAPE_Test_Max[count] = MAPE_Test.max()  
    List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
    print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


figure, axis = plt.subplots(1,2) 
axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")


axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[0].set_title("MSE, Hidden_sigmoid 10, L2_regul")
axis[0].legend(loc="best")



axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[1].set_title("MAPE, Hidden_sigmoid 10, L2_regul")
axis[1].legend(loc="best")


List_MSE_L2_Min = np.copy(List_MSE_Test_Min)
List_MSE_L2_Max = np.copy(List_MSE_Test_Max)
List_MSE_L2_Mean = np.copy(List_MSE_Test_Mean) 
List_MAPE_L2_Min = np.copy(List_MAPE_Test_Min) 
List_MAPE_L2_Max = np.copy(List_MAPE_Test_Max)  
List_MAPE_L2_Mean = np.copy(List_MAPE_Test_Mean)  








for count, reg in enumerate(List_reg):
    for iteration in range(Number_measure):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        # print(iteration, l_feat)
        network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,), kernel_regularizer=regularizers.L1L2(l1 = 10**reg, l2 = 10**reg)))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005, decay = 0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 200,batch_size = 50, validation_split = 0.1)
        if iteration == Number_measure - 1:
            plot_loss(history, string_l = "NN Adam MAE", draw_val = True)
        MSE_Train[iteration] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train[iteration] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test[iteration] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test[iteration] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        keras.backend.clear_session()
    List_MSE_Train_Min[count] = MSE_Train.min() 
    List_MSE_Train_Max[count] = MSE_Train.max() 
    List_MSE_Train_Mean[count] = MSE_Train.mean() 
    List_MAPE_Train_Min[count] = MAPE_Train.min() 
    List_MAPE_Train_Max[count] = MAPE_Train.max() 
    List_MAPE_Train_Mean[count] = MAPE_Train.mean() 

    List_MSE_Test_Min[count] = MSE_Test.min() 
    List_MSE_Test_Max[count] = MSE_Test.max() 
    List_MSE_Test_Mean[count] = MSE_Test.mean() 
    List_MAPE_Test_Min[count] = MAPE_Test.min() 
    List_MAPE_Test_Max[count] = MAPE_Test.max()  
    List_MAPE_Test_Mean[count] = MAPE_Test.mean()  
    print(count, reg, MAPE_Test.min(), MAPE_Test.mean())


figure, axis = plt.subplots(1,2) 
axis[0].plot(List_reg, List_MSE_Train_Min, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Max, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")


axis[0].plot(List_reg, List_MSE_Test_Min, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Max, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[0].set_title("MSE, Hidden_sigmoid 10, L1L2_regul")
axis[0].legend(loc="best")



axis[1].plot(List_reg, List_MAPE_Train_Min, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Max, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_Train_Mean, label="Train_mean", linewidth=2.0, color = "red")

axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[1].plot(List_reg, List_MAPE_Test_Min, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Max, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_Test_Mean, label="Test_Reg_mean", linewidth=2.0, color = "green")

axis[1].set_title("MAPE, Hidden_sigmoid 10, L1L2_regul")
axis[1].legend(loc="best")


List_MSE_L1L2_Min = np.copy(List_MSE_Test_Min)
List_MSE_L1L2_Max = np.copy(List_MSE_Test_Max)
List_MSE_L1L2_Mean = np.copy(List_MSE_Test_Mean) 
List_MAPE_L1L2_Min = np.copy(List_MAPE_Test_Min) 
List_MAPE_L1L2_Max = np.copy(List_MAPE_Test_Max)  
List_MAPE_L1L2_Mean = np.copy(List_MAPE_Test_Mean)  




figure, axis = plt.subplots(1,2)

axis[0].plot(List_reg, List_MSE_non_Min, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Max, linewidth=0.75, color = "black")
axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[0].plot(List_reg, List_MSE_L1_Min, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_L1_Max, linewidth=0.75, color = "red")
axis[0].plot(List_reg, List_MSE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")

axis[0].plot(List_reg, List_MSE_L2_Min, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_L2_Max, linewidth=0.75, color = "green")
axis[0].plot(List_reg, List_MSE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")

axis[0].plot(List_reg, List_MSE_L1L2_Min, linewidth=0.75, color = "blue")
axis[0].plot(List_reg, List_MSE_L1L2_Max, linewidth=0.75, color = "blue")
axis[0].plot(List_reg, List_MSE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")


axis[0].set_title("MSE, Hidden_sigmoid 10")
axis[0].legend(loc="best")



axis[1].plot(List_reg, List_MAPE_non_Min, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Max, linewidth=0.75, color = "black")
axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")

axis[1].plot(List_reg, List_MAPE_L1_Min, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_L1_Max, linewidth=0.75, color = "red")
axis[1].plot(List_reg, List_MAPE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")

axis[1].plot(List_reg, List_MAPE_L2_Min, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_L2_Max, linewidth=0.75, color = "green")
axis[1].plot(List_reg, List_MAPE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")

axis[1].plot(List_reg, List_MAPE_L1L2_Min, linewidth=0.75, color = "blue")
axis[1].plot(List_reg, List_MAPE_L1L2_Max, linewidth=0.75, color = "blue")
axis[1].plot(List_reg, List_MAPE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")



axis[1].set_title("MAPE, Hidden_sigmoid 10")
axis[1].legend(loc="best")







figure, axis = plt.subplots(1,2)

axis[0].plot(List_reg, List_MSE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")
axis[0].plot(List_reg, List_MSE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")
axis[0].plot(List_reg, List_MSE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")
axis[0].plot(List_reg, List_MSE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")

axis[0].set_title("MSE, Hidden_sigmoid 10, Mean")
axis[0].legend(loc="best")

axis[1].plot(List_reg, List_MAPE_non_Mean, label="NonReg_mean", linewidth=2.0, color = "black")
axis[1].plot(List_reg, List_MAPE_L1_Mean, label="L1Reg_mean", linewidth=2.0, color = "red")
axis[1].plot(List_reg, List_MAPE_L2_Mean, label="L2Reg_mean", linewidth=2.0, color = "green")
axis[1].plot(List_reg, List_MAPE_L1L2_Mean, label="L1L2Reg_mean", linewidth=2.0, color = "blue")

axis[1].set_title("MAPE, Hidden_sigmoid 10, Mean")
axis[1].legend(loc="best")





figure, axis = plt.subplots(1,2)

axis[0].plot(List_reg, List_MSE_non_Min, label="NonReg_Min", linewidth=2.0, color = "black")
axis[0].plot(List_reg, List_MSE_L1_Min, label="L1Reg_Min", linewidth=2.0, color = "red")
axis[0].plot(List_reg, List_MSE_L2_Min, label="L2Reg_Min", linewidth=2.0, color = "green")
axis[0].plot(List_reg, List_MSE_L1L2_Min, label="L1L2Reg_Min", linewidth=2.0, color = "blue")

axis[0].set_title("MSE, Hidden_sigmoid 10, Min")
axis[0].legend(loc="best")

axis[1].plot(List_reg, List_MAPE_non_Min, label="NonReg_Min", linewidth=2.0, color = "black")
axis[1].plot(List_reg, List_MAPE_L1_Min, label="L1Reg_Min", linewidth=2.0, color = "red")
axis[1].plot(List_reg, List_MAPE_L2_Min, label="L2Reg_Min", linewidth=2.0, color = "green")
axis[1].plot(List_reg, List_MAPE_L1L2_Min, label="L1L2Reg_Min", linewidth=2.0, color = "blue")

axis[1].set_title("MAPE, Hidden_sigmoid 10, Min")
axis[1].legend(loc="best")




































# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "ML Pol" + str(Degree),  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# plotCoefficients(lr, X_train=X_train_scaled_poly)


#                                                                         # NN Linear Model Adam

# network = models.Sequential()
# l_feat = X_train_scaled_poly.shape[1]
# network.add(layers.Dense(units = 1, activation = 'linear', input_shape=(l_feat,)))
# network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.001), metrics=['mean_squared_error']) 
# history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 500, validation_split = 0)

# def plot_loss(history, string_l = 'Loss', draw_val = False):
#     plt.figure(figsize = (16,10))
#     plt.plot(history.history['loss'], label='loss')
#     if draw_val:
#         plt.plot(history.history['val_loss'], label='val_loss')
#     plt.title(string_l)
#     plt.ylim([0, 10])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [MSE]')   
#     plt.legend()
#     plt.grid(True)

# plot_loss(history, string_l = "NN(ML) Adam 0.001")
# plotModelResults(network, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "NN(ML) Adam 0.001,  Window = 1 day",  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# keras.backend.clear_session()






                                                                        # MSE

# Number_measure = 5

# MSE_Train = np.array([np.NaN] * Number_measure)
# MAPE_Train = np.array([np.NaN] * Number_measure)
# MSE_Test = np.array([np.NaN] * Number_measure)
# MAPE_Test = np.array([np.NaN] * Number_measure)

# activ_func = 'sigmoid'

# loss_func = 'mean_squared_error'

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss= loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MSE) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MSE) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MSE) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MSE) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss= loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN Adadelta MSE", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MSE) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MSE) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MSE) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MSE) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


                                                                # MAE

# loss_func = 'mean_absolute_error'

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MAE) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MAE) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MAE) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MAE) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN Adadelta MAE", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MAE) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MAE) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MAE) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MAE) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


                                                                    # LogCosH

# loss_func = 'log_cosh'

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (LogCosH) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (LogCosH) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (LogCosH) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (LogCosH) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN adadelta LogCosH", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (LogCosH) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (LogCosH) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (LogCosH) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (LogCosH) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


































































# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)
# time_all = time_train.append(time_test, ignore_index=True)


# print(X_all_scaled_poly.head(10))
# print(X_all_scaled_poly['R_CO'][0])
# print(y_all.shape[0])
# Window = np.array([np.NaN] * y_all.shape[0])
# for j in range(y_all.shape[0]):
#     if j > 23:
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j-23, j+1))/24.0
#     else:
#         print(sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1)))
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1))/(j+1.0)                 
# X_all_scaled_poly['window'] = Window
# X_all_scaled_poly['delta'] = X_all_scaled_poly['R_CO'] - X_all_scaled_poly['window']
# plt.figure(figsize = (16,10))
# plt.plot(time_all, X_all_scaled_poly['R_CO'], label = 'R_CO')
# plt.plot(time_all, X_all_scaled_poly['delta'], label = 'delta')
# plt.plot(time_all, X_all_scaled_poly['window'], label = 'window')
# plt.plot(time_all, y_all, label = 'CO')
# # plt.plot(time_all, np.log(y_all), label = 'ln_CO')
# plt.title("Window 24 hour")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)





















# for j in range(y_all.shape[0]):
#     if j > 24*7-1:
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j-(24*7-1), j+1))/(24*7)
#     else:
#         print(sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1)))
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1))/(j+1.0)    
# print(Window)             
# X_all_scaled_poly['window'] = Window
# X_all_scaled_poly['delta'] = X_all_scaled_poly['R_CO'] - X_all_scaled_poly['window']
# print(X_all_scaled_poly)


# plt.figure(figsize = (16,10))
# plt.plot(time_all, X_all_scaled_poly['R_CO'], label = 'R_CO')
# plt.plot(time_all, X_all_scaled_poly['delta'], label = 'delta')
# plt.plot(time_all, X_all_scaled_poly['window'], label = 'window')
# plt.plot(time_all, y_all, label = 'CO')
# plt.title("Window 7 day")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)


# for j in range(y_all.shape[0]):
#     if j > 24*30-1:
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j-(24*30-1), j+1))/(24*30)
#     else:
#         print(sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1)))
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1))/(j+1.0)    
# print(Window)             
# X_all_scaled_poly['window'] = Window
# X_all_scaled_poly['delta'] = X_all_scaled_poly['R_CO'] - X_all_scaled_poly['window']
# print(X_all_scaled_poly)


# plt.figure(figsize = (16,10))
# plt.plot(time_all, X_all_scaled_poly['R_CO'], label = 'R_CO')
# plt.plot(time_all, X_all_scaled_poly['delta'], label = 'delta')
# plt.plot(time_all, X_all_scaled_poly['window'], label = 'window')
# plt.plot(time_all, y_all, label = 'CO')
# plt.title("Window 30 day")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)




# for j in range(y_all.shape[0]):
#     if j > 3:
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j-2, j+1))/4.0
#     else:
#         print(sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1)))
#         Window[j] = sum(X_all_scaled_poly['R_CO'][alpha] for alpha in range(j+1))/(j+1.0)    
# print(Window)             
# X_all_scaled_poly['window'] = Window
# X_all_scaled_poly['delta'] = X_all_scaled_poly['R_CO'] - X_all_scaled_poly['window']
# print(X_all_scaled_poly)


# plt.figure(figsize = (16,10))
# # plt.plot(time_all, X_all_scaled_poly['R_CO'], label = 'R_CO')
# plt.plot(time_all, X_all_scaled_poly['window'], label = 'window')
# # plt.plot(time_all, X_all_scaled_poly['delta'], label = 'delta')
# plt.title("Window 4 hour")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)






# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)

# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)
# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "ML Pol" + str(Degree),  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# plotCoefficients(lr, X_train=X_train_scaled_poly)

# # learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 10, 'hour')



#                                                                         # NN Linear Model Adam

# network = models.Sequential()
# l_feat = X_train_scaled_poly.shape[1]
# network.add(layers.Dense(units = 1, activation = 'linear', input_shape=(l_feat,)))
# network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.001), metrics=['mean_squared_error']) 
# history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0)

# def plot_loss(history, string_l = 'Loss', draw_val = False):
#     plt.figure(figsize = (16,10))
#     plt.plot(history.history['loss'], label='loss')
#     if draw_val:
#         plt.plot(history.history['val_loss'], label='val_loss')
#     plt.title(string_l)
#     plt.ylim([0, 10])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [MSE]')   
#     plt.legend()
#     plt.grid(True)

# plot_loss(history, string_l = "NN(ML) Adam 0.001")
# plotModelResults(network, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "NN(ML) Adam 0.001",  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# keras.backend.clear_session()









#                                                                     # Weights model 

# # # print(network.layers[0].weights)
# # # coefs = pd.DataFrame(X_train_scaled_poly.columns, lr.coef_)
# # # print(coefs)

# # # ann_viz(network, view = True, filename = "fig_NN/network.gv", title="NN")

# # # keras.utils.plot_model(network, to_file='model.png',
# # #     show_shapes=True,
# # #     show_dtype=True,
# # #     show_layer_names=True,
# # #     rankdir='TB',
# # #     expand_nested=True,
# # #     dpi=96,
# # #     layer_range=None,
# # #     show_layer_activations=True
# # # )

# # # test_results = {}
# # # test_results['network_model'] = network.evaluate(X_test_scaled_poly,y_test, verbose=0)
# # # print(test_results)



#                                                 # Error in fit model: Test MSE +- 0.1, MAPE +- 1% 

# # List_Function = ['linear', 'elu', 'relu', 'selu', 'sigmoid', 'tanh', 'exponential']
# # activ_func = 'tanh'
# # activ_func = 'relu'
# # activ_func = 'sigmoid'
# # activ_func = 'exponential'
# # activ_func = 'selu'
# activ_func = 'softmax'

# loss_func = 'mean_squared_error'

# Number_measure = 5

# MSE_Train = np.array([np.NaN] * Number_measure)
# MAPE_Train = np.array([np.NaN] * Number_measure)
# MSE_Test = np.array([np.NaN] * Number_measure)
# MAPE_Test = np.array([np.NaN] * Number_measure)

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss= loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MSE) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MSE) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MSE) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MSE) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss= loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN Adadelta MSE", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MSE) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MSE) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MSE) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MSE) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())




# loss_func = 'mean_absolute_error'

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MAE) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MAE) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MAE) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MAE) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN Adadelta MAE", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (MAE) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (MAE) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (MAE) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (MAE) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())



# loss_func = 'log_cosh'

# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#     # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adam , count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adam, count")
# axis[1].legend(loc="best")

# print("MSE_Train (LogCosH) Adam", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (LogCosH) Adam",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (LogCosH) Adam",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (LogCosH) Adam",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())


# for count in range(Number_measure):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]
#     # print(count, l_feat)
#     network.add(layers.Dense(units = 10, activation = activ_func, input_shape=(l_feat,)))
#     network.add(layers.Dense(units=1, activation = 'linear'))
#     network.compile(loss=loss_func, optimizer= keras.optimizers.Adadelta(learning_rate = 0.1, rho = 0.95), metrics=['mean_squared_error']) 
#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 1000, validation_split = 0.1)
#     plot_loss(history, string_l = "NN adadelta LogCosH", draw_val = True)
#     MSE_Train[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#     MSE_Test[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE Adadelta, count")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE Adadelta, count")
# axis[1].legend(loc="best")

# print("MSE_Train (LogCosH) Adadelta", MSE_Train.mean(), MSE_Train.min(), MSE_Train.max())
# print("MSE_Test (LogCosH) Adadelta",  MSE_Test.mean(), MSE_Test.min(), MSE_Test.max())
# print("MAPE_Train (LogCosH) Adadelta",MAPE_Train.mean(), MAPE_Train.min(), MAPE_Train.max())
# print("MAPE_Test (LogCosH) Adadelta",  MAPE_Test.mean(), MAPE_Test.min(), MAPE_Test.max())














plt.show()
