# Опреление средних значений и квантилей показаний датчика от температуры
# Не было выявлено зависимости значений концентраций от температуры
# Был добавлен столбец часов и выходных
# Рисует графики сопротивлений от температуры
# Исправлена ошибка y_true, prediction

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


df = pd.read_csv('AirQualityUCI/AirQualityUCI.csv', sep = ';') 


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

# feat_CO = ['PT08.S1(CO)','T', 'CO(GT)']
feat_CO = ['PT08.S1(CO)', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)','T','RH', 'CO(GT)']
# лучшие результаты на 28%
feat_target = feat_CO[-1]
l_feat = len(feat_CO) - 1

for feat in feat_CO:
    df_new = df_new[df_new[feat] > -100]

from scipy.optimize import curve_fit

N_bin = 16


T_min = df_new['T'].min()
T_max = df_new['T'].max()
Arr_CO_min = np.array([np.NaN] * N_bin)
Arr_CO_Q1 = np.array([np.NaN] * N_bin)
Arr_CO_Q3 = np.array([np.NaN] * N_bin)
Arr_CO_QX = np.array([np.NaN] * N_bin)
Arr_CO_T = np.array([np.NaN] * N_bin)

# _, axes = plt.subplots(4, 4, sharey = True, figsize = (12, 12))

for k in range(N_bin):
    df_temp = df_new[df_new['T'] >= T_min + (T_max-T_min)*(k/N_bin)]
    df_temp = df_temp[df_new['T'] <= T_min + (T_max-T_min) * ((k+1)/N_bin)]
    Arr_CO_T[k] = T_min + (T_max-T_min) * ((k+0.5)/N_bin)
    Arr_CO_min[k] = df_temp['PT08.S1(CO)'].min()
    Arr_CO_Q1[k]  = df_temp['PT08.S1(CO)'].quantile(0.25)
    Arr_CO_Q3[k]  = df_temp['PT08.S1(CO)'].quantile(0.5)
    Arr_CO_QX[k]  = Arr_CO_Q1[k] - 1.0*(Arr_CO_Q3[k] - Arr_CO_Q1[k])     
    # sns.boxplot(x = 'PT08.S1(CO)', data = df_temp, ax = axes[k//4][k % 4]);
  
def func_exp(x, a, b, c):
    # return a*np.exp(b*x) + c
    return a*(x**2) + b*x + c

plt.figure(figsize=(15, 7))

plt.plot(df_new['T'],df_new['PT08.S1(CO)'],label=str("CO (T)"), marker = 'o', linestyle = 'None', color = "black")

popt_min, pcov_min = curve_fit(func_exp, Arr_CO_T, Arr_CO_min)
# print ("a = %s , b = %s, c = %s" % (popt[0], popt[1], popt[2]))
plt.plot(Arr_CO_T, Arr_CO_min, label = "Min",marker = 'o', linestyle = 'None',  color = "r")
plt.plot(Arr_CO_T, func_exp(Arr_CO_T, *popt_min), label="Fitted Min", color = "r")

df_new['I0_min'] = func_exp(df_new['T'], *popt_min) 

popt_Q1, pcov_Q1 = curve_fit(func_exp, Arr_CO_T, Arr_CO_Q1)
# print ("a = %s , b = %s, c = %s" % (popt[0], popt[1], popt[2]))
plt.plot(Arr_CO_T, Arr_CO_Q1, label = "Q1",marker = 'o', linestyle = 'None', color = "g")
plt.plot(Arr_CO_T, func_exp(Arr_CO_T, *popt_Q1), label="Fitted Q1", color = "g")

df_new['I0_Q1'] = func_exp(df_new['T'], *popt_Q1) 

popt_QX, pcov_QX = curve_fit(func_exp, Arr_CO_T, Arr_CO_QX)
# print ("a = %s , b = %s, c = %s" % (popt[0], popt[1], popt[2]))
plt.plot(Arr_CO_T, Arr_CO_QX, label = "QX",marker = 'o', linestyle = 'None', color = "b")
plt.plot(Arr_CO_T, func_exp(Arr_CO_T, *popt_QX), label="Fitted QX", color = "b")

df_new['I0_QX'] = func_exp(df_new['T'], *popt_QX)


plt.legend(loc="best")

# sns.jointplot(x = 'T', y = 'PT08.S1(CO)', data = df_new, kind = 'scatter')
# sns.jointplot(x = 'T', y = 'PT08.S1(CO)', data = df_new, kind = 'kde', color = "g")


# plt.show()


# df_src_diffmean[feat_float_diff].plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(12,12))
# # plt.savefig('data_fig/density_raw.png')
# # plt.show()

# _, axes = plt.subplots(2, 2, sharey = True, figsize = (12, 12))
# for value, c in enumerate(feat_float_diff): 
#     sns.violinplot(x = c, data = df_src_diffmean, ax = axes[value//2][value % 2]);
# # plt.savefig('data_fig/violin_raw.png')


																		# linear regression

data = pd.DataFrame(df_new[['datetime'] + ['I0_QX'] + feat_CO ].copy())
print(data.tail(7))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_s_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# for time-series cross-validation set 5 folds
tscv = TimeSeriesSplit(n_splits=5)

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    return X_train, X_test, y_train, y_test

y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']
X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)
print(X_train.head(5))
print(y_train.head(5))
print(time_train.head(5))

temp_str = " "
def plotModelResults(
    model, X_train=X_train, X_test=X_test,string = temp_str, plot_intervals=False, plot_anomalies=False, time_test = time_test, time_train = time_train
):
    prediction = model.predict(X_test)
    plt.figure(figsize=(15, 7))
    plt.plot(time_test, prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(time_test, y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
        )
        mae = cv.mean() * (-1)
        deviation = cv.std()
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        plt.plot(time_test, lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(time_test, upper, "r--", alpha=0.5)
        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(time_test, anomalies, "o", markersize=10, label="Anomalies")
    error = mean_absolute_percentage_error(y_test, prediction)
    error_mse = mean_s_error(y_test, prediction)

    plt.title("MAPE: {0:.2f}% ".format(error) + "MSE: {0:.2f} ".format(error_mse) + str(string))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)

def plotCoefficients(model):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed");



# machine learning in two lines
lr = LinearRegression()
lr.fit(X_train, y_train)
plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, string = " scaled", plot_intervals=True)
plotCoefficients(lr)


data["hour"] = df_new['datetime'].dt.hour
data["weekday"] = df_new['datetime'].dt.weekday
# data["is_weekend"] = df_new['datetime'].dt.weekday.isin([5, 6]) * 1
# data.tail()

y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']

X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled,string =" scaled with hour", plot_intervals=True)
plotCoefficients(lr)



from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)

plotModelResults(
    ridge,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    plot_intervals=True,
    plot_anomalies=True,
    string =" scaled Ridge"
)
plotCoefficients(ridge)



lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)

plotModelResults(
    lasso,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    plot_intervals=True,
    plot_anomalies=True,
    string =" scaled Lasso"
)
plotCoefficients(lasso)

                        # BOOSTING

from xgboost import XGBRegressor

xgb = XGBRegressor(verbosity=0)
xgb.fit(X_train_scaled, y_train);

plotModelResults(
    xgb,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    plot_intervals=True,
    plot_anomalies=True,
    string =" scaled XGBR"
)

plt.show()

