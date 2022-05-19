# Построение Тренировочных кривых и функций активаций, вывод: 
# достаточно 2000 измерений для прогнозирования 2000 тысяч данных на основе
# tanh с 5 скрытыми нейронами в 1 слое, валидация 30% выходной линейный слой

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

        # plt.figure(figsize = (16,10))
        # for mon in range(1,13):
        #     plt.plot(mon, temp[temp['Month'] == mon]['GRE'].mean())



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
    

def PolyScaledData(data_df = df_new, target_name = feat_target, pol_deg = Degree, TimeSplit = True, Test_size = T_size, add_hour = False, add_CO_hour = False, list_drop = list_delete):
    if add_hour:
        data["hour"] = data_df['datetime'].dt.hour
    if add_CO_hour:
        data["hour"] = data_df['datetime'].dt.hour
        MeanCo_hour = np.array([np.NaN] * 24)
        ar_hour = range(24)
        for i in range(24):
            MeanCo_hour[i] = data[data['hour'] == i]['CO(GT)'].mean()
        data["COmean"] = MeanCo_hour[data["hour"]]

  
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
        COmean_train_scaled = X_train_scaled.dropna()['COmean']
        COmean_test_scaled = X_test_scaled.dropna()['COmean']
        X_train_scaled = X_train_scaled.dropna().drop(['COmean'], axis=1)
        X_test_scaled = X_test_scaled.dropna().drop(['COmean'], axis=1)


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
        X_train_scaled_poly['COmean'] = COmean_train_scaled 
        X_test_scaled_poly['COmean'] = COmean_test_scaled


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

X_train_scaled_poly, y_train, time_train, X_test_scaled_poly, y_test, time_test = PolyScaledData(add_hour = True)

X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
y_all = y_train.append(y_test, ignore_index=True)


lr = LinearRegression()
lr.fit(X_train_scaled_poly, y_train)
plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "ML Pol" + str(Degree),  plot_intervals=True,\
                 y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
plotCoefficients(lr, X_train=X_train_scaled_poly)

# learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 10, 'hour')



                                                                        # NN Linear Model

network = models.Sequential()
l_feat = X_train_scaled_poly.shape[1]

network.add(layers.Dense(units = 1, activation = 'linear', input_shape=(l_feat,)))
network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.0001), metrics=['mean_squared_error']) 

history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 500, validation_split = 0)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.head(5))
# print(hist.tail(5))

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

plot_loss(history, string_l = "NN linear regression")

plotModelResults(network, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "NN hour + pol:"+ str(Degree),  plot_intervals=True,\
                 y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)

# print(network.layers[0].weights)
# coefs = pd.DataFrame(X_train_scaled_poly.columns, lr.coef_)
# print(coefs)

# ann_viz(network, view = True, filename = "fig_NN/network.gv", title="NN")

# keras.utils.plot_model(network, to_file='model.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True
# )
keras.backend.clear_session()

# test_results = {}
# test_results['network_model'] = network.evaluate(X_test_scaled_poly,y_test, verbose=0)
# print(test_results)


                                                                        # Neurons: 5 - 15, optimal - 10


List_Neur = range(1, 25, 2)

MSE_Train_Neur = np.array([np.NaN] * len(List_Neur))
MAPE_Train_Neur = np.array([np.NaN] * len(List_Neur))
MSE_Test_Neur = np.array([np.NaN] * len(List_Neur))
MAPE_Test_Neur = np.array([np.NaN] * len(List_Neur))

MSE_Train_temp =  np.array([np.NaN] * 3)
MAPE_Train_temp =  np.array([np.NaN] * 3)
MSE_Test_temp =  np.array([np.NaN] * 3)
MAPE_Test_temp =  np.array([np.NaN] * 3)


for count, neur in enumerate(List_Neur):
    for j in range(3):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        network.add(layers.Dense(units = neur, activation = 'tanh', input_shape=(l_feat,)))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
        # plot_loss(history, string_l = "NN tanh" + str(neur))
        MSE_Train_temp[j] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train_temp[j] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test_temp[j] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test_temp[j] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        keras.backend.clear_session()
    print(count,str("Neurons: "), neur)
    MSE_Train_Neur[count] = MSE_Train_temp.mean()
    MAPE_Train_Neur[count] = MAPE_Train_temp.mean()
    MSE_Test_Neur[count] = MSE_Test_temp.mean()
    MAPE_Test_Neur[count] = MAPE_Test_temp.mean() 

figure, axis = plt.subplots(1,2) 
axis[0].plot(List_Neur, MSE_Train_Neur, label="train", linewidth=2.0, color = "red")
axis[0].plot(List_Neur, MSE_Test_Neur, label="test", linewidth=2.0, color = "green")
axis[0].set_title("MSE, Neurons")
axis[0].legend(loc="best")
axis[1].plot(List_Neur, MAPE_Train_Neur, label="train", linewidth=2.0, color = "orange")
axis[1].plot(List_Neur, MAPE_Test_Neur, label="test", linewidth=2.0, color = "dodgerblue")
axis[1].set_title("MAPE, Neurons")
axis[1].legend(loc="best")



                                                                        # Speed Adam: 0.001 - 0.01, optimal 0.005



# List_Speed = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

# MSE_Train = np.array([np.NaN] * len(List_Speed))
# MAPE_Train = np.array([np.NaN] * len(List_Speed))
# MSE_Test = np.array([np.NaN] * len(List_Speed))
# MAPE_Test = np.array([np.NaN] * len(List_Speed))

# for count, speed in enumerate(List_Speed):
#     for j in range(3):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]

#         network.add(layers.Dense(units = 10, activation = 'tanh', input_shape=(l_feat,)))
#         network.add(layers.Dense(units=1, activation = 'linear'))

#         network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(speed), metrics=['mean_squared_error']) 

#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
#         # plot_loss(history, string_l = "NN tanh" + str(count), draw_val = True)
#         MSE_Train_temp[j] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train_temp[j] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test_temp[j] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test_temp[j] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     print(count,str("speed Adam: "), speed)    
#     MSE_Train[count] = MSE_Train_temp.mean()
#     MAPE_Train[count] = MAPE_Train_temp.mean()
#     MSE_Test[count] = MSE_Test_temp.mean()
#     MAPE_Test[count] = MAPE_Test_temp.mean() 


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_Speed, MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(List_Speed, MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE, speed")
# axis[0].legend(loc="best")
# axis[1].plot(List_Speed, MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(List_Speed, MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE, speed")
# axis[1].legend(loc="best")

# figure, axis = plt.subplots(1,2)
# axis[0].plot(MSE_Train, label="train", linewidth=2.0, color = "red")
# axis[0].plot(MSE_Test, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE, speed")
# axis[0].legend(loc="best")
# axis[1].plot(MAPE_Train, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(MAPE_Test, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE, speed")
# axis[1].legend(loc="best")



                                                                        # Validation: 10 - 30 %

# List_Validation = np.arange(0.0, 0.9, 0.02)

# MSE_Train_Validation = np.array([np.NaN] * len(List_Validation))
# MAPE_Train_Validation = np.array([np.NaN] * len(List_Validation))
# MSE_Test_Validation = np.array([np.NaN] * len(List_Validation))
# MAPE_Test_Validation = np.array([np.NaN] * len(List_Validation))

# MSE_Train_temp =  np.array([np.NaN] * 3)
# MAPE_Train_temp =  np.array([np.NaN] * 3)
# MSE_Test_temp =  np.array([np.NaN] * 3)
# MAPE_Test_temp =  np.array([np.NaN] * 3)


# for count, validation in enumerate(List_Validation):
#     for j in range(3):
#         network = models.Sequential()
#         l_feat = X_train_scaled_poly.shape[1]

#         network.add(layers.Dense(units = 10, activation = 'tanh', input_shape=(l_feat,)))
#         network.add(layers.Dense(units=1, activation = 'linear'))

#         network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 

#         history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = validation)
#         # plot_loss(history, string_l = "NN tanh" + str(validation))
#         MSE_Train_temp[j] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#         MAPE_Train_temp[j] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
#         MSE_Test_temp[j] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#         MAPE_Test_temp[j] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
#         keras.backend.clear_session()
#     print(count,str("validation: "), validation)
#     MSE_Train_Validation[count] = MSE_Train_temp.mean()
#     MAPE_Train_Validation[count] = MAPE_Train_temp.mean()
#     MSE_Test_Validation[count] = MSE_Test_temp.mean()
#     MAPE_Test_Validation[count] = MAPE_Test_temp.mean() 

# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_Validation, MSE_Train_Validation, label="train", linewidth=2.0, color = "red")
# axis[0].plot(List_Validation, MSE_Test_Validation, label="test", linewidth=2.0, color = "green")
# axis[0].set_title("MSE, validation")
# axis[0].legend(loc="best")
# axis[1].plot(List_Validation, MAPE_Train_Validation, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(List_Validation, MAPE_Test_Validation, label="test", linewidth=2.0, color = "dodgerblue")
# axis[1].set_title("MAPE, validation")
# axis[1].legend(loc="best")












                                                           # Hidden layer(5:5...5:5) - optimal 1 


List_Layer = range(1, 6, 1)

MSE_Train_Layer = np.array([np.NaN] * len(List_Layer))
MAPE_Train_Layer = np.array([np.NaN] * len(List_Layer))
MSE_Test_Layer = np.array([np.NaN] * len(List_Layer))
MAPE_Test_Layer = np.array([np.NaN] * len(List_Layer))

for count, layer in enumerate(List_Layer):
    for j in range(3):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        network.add(layers.Dense(units = 5, activation = 'tanh', input_shape=(l_feat,)))
        for k in range(1, layer):
            network.add(layers.Dense(units = 5, activation = 'tanh'))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
        if j == 2:
            plot_loss(history, string_l = "NN tanh with Hidden Layer:" + str(layer))
        MSE_Train_temp[j] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train_temp[j] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test_temp[j] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test_temp[j] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        # ann_viz(network, view = True, filename = "fig_NN/NN_HiddenLayer" + str(layer) + ".gv", title="NN, Hidden" + str(layer))
        keras.backend.clear_session()
    print(count,str("Layer: "), layer)    
    MSE_Train_Layer[count] = MSE_Train_temp.mean()
    MAPE_Train_Layer[count] = MAPE_Train_temp.mean()
    MSE_Test_Layer[count] = MSE_Test_temp.mean()
    MAPE_Test_Layer[count] = MAPE_Test_temp.mean() 

figure, axis = plt.subplots(1,2) 
axis[0].plot(List_Layer, MSE_Train_Layer, label="train", linewidth=2.0, color = "red")
axis[0].plot(List_Layer, MSE_Test_Layer, label="test", linewidth=2.0, color = "green")
axis[0].set_title("MSE, hidden layer")
axis[0].legend(loc="best")
axis[1].plot(List_Layer, MAPE_Train_Layer, label="train", linewidth=2.0, color = "orange")
axis[1].plot(List_Layer, MAPE_Test_Layer, label="test", linewidth=2.0, color = "dodgerblue")
axis[1].set_title("MAPE, hidden layer")
axis[1].legend(loc="best")


                                                         # Hidden layer(5:2^n...2:1) - optimal  2/4


for count, layer in enumerate(List_Layer):
    for j in range(3):
        network = models.Sequential()
        l_feat = X_train_scaled_poly.shape[1]
        network.add(layers.Dense(units = 2**layer, activation = 'tanh', input_shape=(l_feat,)))
        for k in range(1, layer):
            network.add(layers.Dense(units = 2**(layer - k), activation = 'tanh'))
        network.add(layers.Dense(units=1, activation = 'linear'))
        network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.005), metrics=['mean_squared_error']) 
        history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)
        if j == 2:
            plot_loss(history, string_l = "NN triangle With hidden Layer:" + str(layer))
        MSE_Train_temp[j] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
        MAPE_Train_temp[j] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())
        MSE_Test_temp[j] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
        MAPE_Test_temp[j] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())
        if j == 2:
            ann_viz(network, view = True, filename = "fig_NN/NN_HiddenLayer_Triangle" + str(layer) + ".gv", title="NN, Hidden Triangle" + str(layer))
        keras.backend.clear_session()
    print(count,str("Layer: "), layer)    
    MSE_Train_Layer[count] = MSE_Train_temp.mean()
    MAPE_Train_Layer[count] = MAPE_Train_temp.mean()
    MSE_Test_Layer[count] = MSE_Test_temp.mean()
    MAPE_Test_Layer[count] = MAPE_Test_temp.mean() 

figure, axis = plt.subplots(1,2) 
axis[0].plot(List_Layer, MSE_Train_Layer, label="train", linewidth=2.0, color = "red")
axis[0].plot(List_Layer, MSE_Test_Layer, label="test", linewidth=2.0, color = "green")
axis[0].set_title("MSE, triangle")
axis[0].legend(loc="best")
axis[1].plot(List_Layer, MAPE_Train_Layer, label="train", linewidth=2.0, color = "orange")
axis[1].plot(List_Layer, MAPE_Test_Layer, label="test", linewidth=2.0, color = "dodgerblue")
axis[1].set_title("MAPE, triangle")
axis[1].legend(loc="best")











# for count, layer in enumerate(List_Layer):
#     network = models.Sequential()
#     l_feat = X_train_scaled_poly.shape[1]

#     network.add(layers.Dense(units = 3**layer, activation = 'tanh', input_shape=(l_feat,)))
   
#     for j in range(1, layer):
#         network.add(layers.Dense(units = 3**(layer - j), activation = 'tanh'))

#     network.add(layers.Dense(units=1, activation = 'linear'))

#     network.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adam(0.001), metrics=['mean_squared_error']) 

#     history = network.fit(X_train_scaled_poly, y_train, verbose=0, epochs = 100, validation_split = 0.1)


#     plot_loss(history, string_l = "NN triangle tanh" + str(layer))

#     MSE_Train_Layer[count] = mean_s_error(y_train,network.predict(X_train_scaled_poly).flatten())    
#     MAPE_Train_Layer[count] = mean_absolute_percentage_error(y_train,network.predict(X_train_scaled_poly).flatten())

#     MSE_Test_Layer[count] = mean_s_error(y_test,network.predict(X_test_scaled_poly).flatten())    
#     MAPE_Test_Layer[count] = mean_absolute_percentage_error(y_test,network.predict(X_test_scaled_poly).flatten())

#     ann_viz(network, view = True, filename = "fig_NN/NN_HiddenLayer_Triangle" + str(layer) + ".gv", title="NN, Hidden Triangle" + str(layer))


#     keras.backend.clear_session()


# figure, axis = plt.subplots(1,2) 
# axis[0].plot(List_Layer, MSE_Train_Layer, label="train", linewidth=2.0, color = "red")
# axis[0].plot(List_Layer, MSE_Test_Layer, label="test", linewidth=2.0, color = "green")

# axis[0].set_title("MSE, val = 0.1")
# axis[0].legend(loc="best")


# axis[1].plot(List_Layer, MAPE_Train_Layer, label="train", linewidth=2.0, color = "orange")
# axis[1].plot(List_Layer, MAPE_Test_Layer, label="test", linewidth=2.0, color = "dodgerblue")

# axis[1].set_title("MAPE, val = 0.1")
# axis[1].legend(loc="best")








                                                            #  Mean CO


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

# X_train_scaled_poly['COmean'] = COmean_train_scaled 
# X_test_scaled_poly['COmean'] = COmean_test_scaled

# X_all_scaled_poly = X_train_scaled_poly.append(X_test_scaled_poly, ignore_index=True) 
# y_all = y_train.append(y_test, ignore_index=True)




# lr = LinearRegression()
# lr.fit(X_train_scaled_poly, y_train)

# plotModelResults(lr, X_train=X_train_scaled_poly, X_test=X_test_scaled_poly, string = "ML Pol" + str(Degree),  plot_intervals=True,\
#                  y_train = y_train, y_test = y_test, time_train = time_train, time_test = time_test, plot_diff = False)
# # plotCoefficients(lr, X_train=X_train_scaled_poly)

# # learning_curves(LinearRegression(), X_all_scaled_poly, y_all,List_Train_Size, 10, 'MeanH_P2')




plt.show()
