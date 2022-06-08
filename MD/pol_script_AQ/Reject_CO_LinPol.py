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


Degree = 2
# list_delete = []
list_delete = ['T^2','R_NM^2']
# list_delete = ['T^2','PT08.S2(NMHC)^2']
T_size = 0.9




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

    prediction = model.predict(X_test)

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
    # print(type(array_mse), len(array_mse))
    array_mape = (array_diff/y_test.values) * 100
    array_gre  = arr_gate_rate_error(y_test.values,prediction)

    # print(array_mse.mean())
    # print(array_mse.max())
    # print(array_mse.min())
    # print(np.sort(array_mse))

    error_mse = mean_s_error(y_test.values,prediction)
    error_gre = gate_rate_error(y_test.values,prediction)

    plt.title("MAPE: {0:.2f}% ".format(error_mape) + "MSE: {0:.2f} ".format(error_mse) + "GRE(25%): {0:.2f}% ".format(error_gre) + str(string))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)


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
    """
        Plots sorted coefficient values of the model
    """

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
    




    







List_Train_Size = np.arange(0.01,1.0,0.01)

# df_new = df_new[df_new['CO(GT)'] > 0.4]
# df_new = df_new[df_new['CO(GT)'] < 5.0]


List_Rejection = np.arange(0.1,2.1,0.1)
print(List_Rejection)

MSE_Reject = np.array([np.NaN] * len(List_Rejection))
MAPE_Reject = np.array([np.NaN] * len(List_Rejection))
GRE_Reject = np.array([np.NaN] * len(List_Rejection))

MSE_Test = np.array([np.NaN] * len(List_Rejection))
MAPE_Test = np.array([np.NaN] * len(List_Rejection))
GRE_Test = np.array([np.NaN] * len(List_Rejection))


MSE_TR = np.array([np.NaN] * len(List_Rejection))
MAPE_TR = np.array([np.NaN] * len(List_Rejection))
GRE_TR = np.array([np.NaN] * len(List_Rejection))






for count, low_lim in enumerate(List_Rejection):
    df_low = df_new[df_new['CO(GT)'] >= low_lim]
    df_reject = df_new[df_new['CO(GT)'] < low_lim]

    print(df_reject.shape[0], df_low.shape[0], df_new.shape[0])
    print(df_reject.shape[0]/df_new.shape[0]*100, df_low.shape[0]/df_new.shape[0]*100)

    data = pd.DataFrame(df_low[['datetime'] + feat_CO].copy())
    data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

    data["hour"] = df_low['datetime'].dt.hour

    data_reject = pd.DataFrame(df_reject[['datetime'] + feat_CO].copy())
    data_reject.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

    data_reject["hour"] = df_reject['datetime'].dt.hour





    y = data.dropna()[feat_target]
    X = data.dropna().drop([feat_target], axis=1)

    y_reject = data_reject.dropna()[feat_target]
    X_reject = data_reject.dropna().drop([feat_target], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 2000)

    # X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)


    time_train = X_train.dropna()['datetime']
    time_test = X_test.dropna()['datetime']
    X_train = X_train.dropna().drop(['datetime'], axis=1)
    X_test = X_test.dropna().drop(['datetime'], axis=1)

    time_reject = X_reject.dropna()['datetime']
    X_reject = X_reject.dropna().drop(['datetime'], axis=1)


    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    list_X = X_test.columns.values.tolist()
    X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
    X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

    if low_lim > 0.1:
        X_reject_scaled = pd.DataFrame(scaler.transform(X_reject))
        X_reject_scaled.set_axis(list_X, axis = 'columns', inplace=True)

    # print(X_reject_scaled)
    # print(X_test_scaled)


    hour_train_scaled = X_train_scaled.dropna()['hour']
    hour_test_scaled = X_test_scaled.dropna()['hour']
    X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
    X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)

    if low_lim > 0.1:
        hour_reject_scaled = X_reject_scaled.dropna()['hour']
        X_reject_scaled = X_reject_scaled.dropna().drop(['hour'], axis=1)
 

    poly = PolynomialFeatures(degree=Degree)


    X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
    X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
    list_scaled_poly =list(poly.get_feature_names_out())
    X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
    X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
    X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
    X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

    if low_lim > 0.1:
        X_reject_scaled_poly = pd.DataFrame(poly.fit_transform(X_reject_scaled))
        X_reject_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
        X_reject_scaled_poly = X_reject_scaled_poly.dropna().drop(list_delete, axis=1)


    X_train_scaled_poly['hour'] = hour_train_scaled
    X_test_scaled_poly['hour'] = hour_test_scaled
    X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
    X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)

    if low_lim > 0.1:
        X_reject_scaled_poly['hour'] = hour_reject_scaled
        X_reject_scaled_poly = X_reject_scaled_poly.dropna().drop(['1'], axis=1)


    # X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
    # X_all_scaled_poly = X_all_scaled_poly.append(X_reject_scaled_poly, ignore_index=True) 

    # y_all = y_train.append(y_test, ignore_index=True)
    # y_all = y_all.append(y_reject, ignore_index=True)

    if low_lim > 0.1:
        X_TR_scaled_poly = X_test_scaled_poly.append(X_reject_scaled_poly, ignore_index=True) 
        y_TR = y_test.append(y_reject, ignore_index=True)



    lr = LinearRegression()
    lr.fit(X_train_scaled_poly, y_train)
    # plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "hour_P" + str(Degree) + "_04",  plot_intervals=True,\
    #                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = True)
    # plotCoefficients(lr, X_train=X_train_scaled_poly)

    predict_test = lr.predict(X_test_scaled_poly)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test.values,predict_test)
    MSE_Test[count] = mean_s_error(y_test.values,predict_test)
    GRE_Test[count] = gate_rate_error(y_test.values,predict_test)

    if low_lim > 0.1:        
        predict_reject = lr.predict(X_reject_scaled_poly)
        predict_TR = lr.predict(X_TR_scaled_poly)
        MAPE_Reject[count] = mean_absolute_percentage_error(y_reject.values,predict_reject)
        MAPE_TR[count] = mean_absolute_percentage_error(y_TR.values, predict_TR)
        MSE_Reject[count] = mean_s_error(y_reject.values,predict_reject)
        MSE_TR[count] = mean_s_error(y_TR.values, predict_TR)
        GRE_Reject[count] = gate_rate_error(y_reject.values,predict_reject)
        GRE_TR[count] = gate_rate_error(y_TR.values,predict_TR)    



plt.figure(figsize=(15, 7))    

plt.axhline(y = MSE_Test[0], label = 'no Rej', color = 'black')       
plt.plot(List_Rejection, MSE_Test, label = 'Test', color = 'red')
plt.plot(List_Rejection, MSE_Reject, label = 'Reject', color = 'green')
plt.plot(List_Rejection, MSE_TR, label = 'Test+Reject', color = 'blue')
plt.title("MSE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = MAPE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, MAPE_Test, label = 'Test', color = 'red')
plt.plot(List_Rejection, MAPE_Reject, label = 'Reject', color = 'green')
plt.plot(List_Rejection, MAPE_TR, label = 'Test+Reject', color = 'blue')
plt.title("MAPE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = GRE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, GRE_Test, label = 'Test', color = 'red')
plt.plot(List_Rejection, GRE_Reject, label = 'Reject', color = 'green')
plt.plot(List_Rejection, GRE_TR, label = 'Test+Reject', color = 'blue')
plt.title("GRE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)
















List_Rejection = np.arange(0.1,3.1,0.1)
print(List_Rejection)


MSE_Test = np.array([np.NaN] * len(List_Rejection))
MAPE_Test = np.array([np.NaN] * len(List_Rejection))
GRE_Test = np.array([np.NaN] * len(List_Rejection))





# df_new = df_new.sample(frac=1).reset_index(drop=True)


for count, low_lim in enumerate(List_Rejection):


    data = pd.DataFrame(df_new[['datetime'] + feat_CO].copy())
    data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

    data["hour"] = df_new['datetime'].dt.hour

    y = data.dropna()[feat_target]
    X = data.dropna().drop([feat_target], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 2000)
    # X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)
    
    X_train[feat_target] = y_train

    X_train = X_train[X_train[feat_target] + 0.01 > low_lim]

    y_train = X_train.dropna()[feat_target]
    X_train = X_train.dropna().drop([feat_target], axis=1)

    print(str(count) + ") low_lim:" + str(low_lim) + " size: " + str(X_train.shape[0]))
    print(y_train.min())
    # print("min y_train = " + str(y_train[feat_target].min()))
    # print(X_train.tail(2))
    # print(X_test.shape[0])

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
    X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
    X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)



    lr = LinearRegression()
    lr.fit(X_train_scaled_poly, y_train)
    # plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "hour_P" + str(Degree) + "_04",  plot_intervals=True,\
    #                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = True)
    # plotCoefficients(lr, X_train=X_train_scaled_poly)

    predict_test = lr.predict(X_test_scaled_poly)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test.values,predict_test)
    MSE_Test[count] = mean_s_error(y_test.values,predict_test)
    GRE_Test[count] = gate_rate_error(y_test.values,predict_test)



plt.figure(figsize=(15, 7))    

plt.axhline(y = MSE_Test[0], label = 'no Rej', color = 'black')       
plt.plot(List_Rejection, MSE_Test, label = 'Test', color = 'red')
plt.title("MSE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = MAPE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, MAPE_Test, label = 'Test', color = 'red')
plt.title("MAPE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = GRE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, GRE_Test, label = 'Test', color = 'red')
plt.title("GRE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)










List_Rejection = np.arange(0.1,10.1,0.1)
print(List_Rejection)


MSE_Test = np.array([np.NaN] * len(List_Rejection))
MAPE_Test = np.array([np.NaN] * len(List_Rejection))
GRE_Test = np.array([np.NaN] * len(List_Rejection))


for count, low_lim in enumerate(List_Rejection):


    data = pd.DataFrame(df_new[['datetime'] + feat_CO].copy())
    data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

    data["hour"] = df_new['datetime'].dt.hour

    y = data.dropna()[feat_target]
    X = data.dropna().drop([feat_target], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 2000)
    # X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)
    
    X_train[feat_target] = y_train

    X_train = X_train[X_train[feat_target] < 12.01 - low_lim]

    y_train = X_train.dropna()[feat_target]
    X_train = X_train.dropna().drop([feat_target], axis=1)

    print(str(count) + ") low_max:" + str(12 - low_lim) + " size: " + str(X_train.shape[0]))
    print(y_train.max())
    # print("min y_train = " + str(y_train[feat_target].min()))
    # print(X_train.tail(2))
    # print(X_test.shape[0])

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
    X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
    X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)



    lr = LinearRegression()
    lr.fit(X_train_scaled_poly, y_train)
    # plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "hour_P" + str(Degree) + "_04",  plot_intervals=True,\
    #                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = True)
    # plotCoefficients(lr, X_train=X_train_scaled_poly)

    predict_test = lr.predict(X_test_scaled_poly)
    MAPE_Test[count] = mean_absolute_percentage_error(y_test.values,predict_test)
    MSE_Test[count] = mean_s_error(y_test.values,predict_test)
    GRE_Test[count] = gate_rate_error(y_test.values,predict_test)



plt.figure(figsize=(15, 7))    

plt.axhline(y = MSE_Test[0], label = 'no Rej', color = 'black')       
plt.plot(List_Rejection, MSE_Test, label = 'Test', color = 'red')
plt.title("MSE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = MAPE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, MAPE_Test, label = 'Test', color = 'red')
plt.title("MAPE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)


plt.figure(figsize=(15, 7))        
plt.axhline(y = GRE_Test[0], label = 'no Rej', color = 'black')
plt.plot(List_Rejection, GRE_Test, label = 'Test', color = 'red')
plt.title("GRE (low_lim)")
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)

































# List_Rejection = np.arange(0.0,3.0,0.1)

# MSE_Reject = np.array([np.NaN] * len(List_Rejection))
# MAPE_Reject = np.array([np.NaN] * len(List_Rejection))
# GRE_Reject = np.array([np.NaN] * len(List_Rejection))

# MSE_Test = np.array([np.NaN] * len(List_Rejection))
# MAPE_Test = np.array([np.NaN] * len(List_Rejection))
# GRE_Test = np.array([np.NaN] * len(List_Rejection))


# MSE_TR = np.array([np.NaN] * len(List_Rejection))
# MAPE_TR = np.array([np.NaN] * len(List_Rejection))
# GRE_TR = np.array([np.NaN] * len(List_Rejection))





# for count, upp_lim in enumerate(List_Rejection):
#     df_upp = df_new[df_new['CO(GT)'] < 7.0 - upp_lim]
#     df_reject = df_new[df_new['CO(GT)'] >= 7.0 - upp_lim]

#     print(df_reject.shape[0], df_upp.shape[0], df_new.shape[0])
#     print(df_reject.shape[0]/df_new.shape[0]*100, df_upp.shape[0]/df_new.shape[0]*100)

#     data = pd.DataFrame(df_upp[['datetime'] + feat_CO].copy())
#     data.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

#     data["hour"] = df_upp['datetime'].dt.hour

#     data_reject = pd.DataFrame(df_reject[['datetime'] + feat_CO].copy())
#     data_reject.rename(columns = {'PT08.S1(CO)':'R_CO', 'PT08.S2(NMHC)':'R_NM'}, inplace = True)

#     data_reject["hour"] = df_reject['datetime'].dt.hour





#     y = data.dropna()[feat_target]
#     X = data.dropna().drop([feat_target], axis=1)

#     y_reject = data_reject.dropna()[feat_target]
#     X_reject = data_reject.dropna().drop([feat_target], axis=1)

#     X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 2000)

#     # X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)


#     time_train = X_train.dropna()['datetime']
#     time_test = X_test.dropna()['datetime']
#     X_train = X_train.dropna().drop(['datetime'], axis=1)
#     X_test = X_test.dropna().drop(['datetime'], axis=1)

#     time_reject = X_reject.dropna()['datetime']
#     X_reject = X_reject.dropna().drop(['datetime'], axis=1)


#     X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
#     X_test_scaled = pd.DataFrame(scaler.transform(X_test))
#     list_X = X_test.columns.values.tolist()
#     X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
#     X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

#     # if low_lim > 0:
#     X_reject_scaled = pd.DataFrame(scaler.transform(X_reject))
#     X_reject_scaled.set_axis(list_X, axis = 'columns', inplace=True)

#     # print(X_reject_scaled)
#     # print(X_test_scaled)


#     hour_train_scaled = X_train_scaled.dropna()['hour']
#     hour_test_scaled = X_test_scaled.dropna()['hour']
#     X_train_scaled = X_train_scaled.dropna().drop(['hour'], axis=1)
#     X_test_scaled = X_test_scaled.dropna().drop(['hour'], axis=1)

#     # if low_lim > 0:
#     hour_reject_scaled = X_reject_scaled.dropna()['hour']
#     X_reject_scaled = X_reject_scaled.dropna().drop(['hour'], axis=1)
 

#     poly = PolynomialFeatures(degree=Degree)


#     X_train_scaled_poly = pd.DataFrame(poly.fit_transform(X_train_scaled))
#     X_test_scaled_poly = pd.DataFrame(poly.fit_transform(X_test_scaled))
#     list_scaled_poly =list(poly.get_feature_names_out())
#     X_train_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
#     X_test_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
#     X_train_scaled_poly = X_train_scaled_poly.dropna().drop(list_delete, axis=1)
#     X_test_scaled_poly  = X_test_scaled_poly.dropna().drop(list_delete, axis=1)

#     # if low_lim > 0:
#     X_reject_scaled_poly = pd.DataFrame(poly.fit_transform(X_reject_scaled))
#     X_reject_scaled_poly.set_axis(list_scaled_poly, axis = 'columns', inplace=True)
#     X_reject_scaled_poly = X_reject_scaled_poly.dropna().drop(list_delete, axis=1)


#     X_train_scaled_poly['hour'] = hour_train_scaled
#     X_test_scaled_poly['hour'] = hour_test_scaled
#     X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
#     X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)

#     # if low_lim > 0:
#     X_reject_scaled_poly['hour'] = hour_reject_scaled
#     X_reject_scaled_poly = X_reject_scaled_poly.dropna().drop(['1'], axis=1)


#     # X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
#     # X_all_scaled_poly = X_all_scaled_poly.append(X_reject_scaled_poly, ignore_index=True) 

#     # y_all = y_train.append(y_test, ignore_index=True)
#     # y_all = y_all.append(y_reject, ignore_index=True)

#     # if low_lim > 0:
#     X_TR_scaled_poly = X_test_scaled_poly.append(X_reject_scaled_poly, ignore_index=True) 
#     y_TR = y_test.append(y_reject, ignore_index=True)



#     lr = LinearRegression()
#     lr.fit(X_train_scaled_poly, y_train)
#     # plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "hour_P" + str(Degree) + "_04",  plot_intervals=True,\
#     #                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = True)
#     # plotCoefficients(lr, X_train=X_train_scaled_poly)

#     predict_test = lr.predict(X_test_scaled_poly)
#     MAPE_Test[count] = mean_absolute_percentage_error(y_test.values,predict_test)
#     MSE_Test[count] = mean_s_error(y_test.values,predict_test)
#     GRE_Test[count] = gate_rate_error(y_test.values,predict_test)

#     # if low_lim > 0:        
#     predict_reject = lr.predict(X_reject_scaled_poly)
#     predict_TR = lr.predict(X_TR_scaled_poly)
#     MAPE_Reject[count] = mean_absolute_percentage_error(y_reject.values,predict_reject)
#     MAPE_TR[count] = mean_absolute_percentage_error(y_TR.values, predict_TR)
#     MSE_Reject[count] = mean_s_error(y_reject.values,predict_reject)
#     MSE_TR[count] = mean_s_error(y_TR.values, predict_TR)
#     GRE_Reject[count] = gate_rate_error(y_reject.values,predict_reject)
#     GRE_TR[count] = gate_rate_error(y_TR.values,predict_TR)    



# plt.figure(figsize=(15, 7))    

# plt.axhline(y = MSE_Test[0], label = 'no Rej', color = 'black')       
# plt.plot(List_Rejection, MSE_Test, label = 'Test', color = 'red')
# plt.plot(List_Rejection, MSE_Reject, label = 'Reject', color = 'green')
# plt.plot(List_Rejection, MSE_TR, label = 'Test+Reject', color = 'blue')
# plt.title("MSE (7 - upp_lim)")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)


# plt.figure(figsize=(15, 7))        
# plt.axhline(y = MAPE_Test[0], label = 'no Rej', color = 'black')
# plt.plot(List_Rejection, MAPE_Test, label = 'Test', color = 'red')
# plt.plot(List_Rejection, MAPE_Reject, label = 'Reject', color = 'green')
# plt.plot(List_Rejection, MAPE_TR, label = 'Test+Reject', color = 'blue')
# plt.title("MAPE (7 - upp_lim)")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)


# plt.figure(figsize=(15, 7))        
# plt.axhline(y = GRE_Test[0], label = 'no Rej', color = 'black')
# plt.plot(List_Rejection, GRE_Test, label = 'Test', color = 'red')
# plt.plot(List_Rejection, GRE_Reject, label = 'Reject', color = 'green')
# plt.plot(List_Rejection, GRE_TR, label = 'Test+Reject', color = 'blue')
# plt.title("GRE (7 - upp_lim)")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.grid(True)

























































# learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 10, 'hour_P2_04')











# data["hour"] = df_new['datetime'].dt.hour


# MeanCo_hour = np.array([np.NaN] * 24)
# ar_hour = range(24)

# for i in range(24):
#     MeanCo_hour[i] = data[data['hour'] == i]['CO(GT)'].mean()


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


# X_train_scaled_poly = X_train_scaled_poly.dropna().drop(['1'], axis=1)
# X_test_scaled_poly = X_test_scaled_poly.dropna().drop(['1'], axis=1)



# # X_train_scaled_poly['hour'] = hour_train_scaled
# # X_test_scaled_poly['hour'] = hour_test_scaled


# X_train_scaled_poly['COmean'] = COmean_train_scaled 
# X_test_scaled_poly['COmean'] = COmean_test_scaled

# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)









# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)

# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "MeanH_P" + str(Degree) + "_04",  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = True)
# plotCoefficients(lr, X_train=X_train_scaled_poly)


# learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 10, 'MeanH_P2_04')






plt.show()