# Постройка и сравнение полиномиальных моделей
# Со списком удалённых столбцов, настраиваемая степень
# Сравнение с добавлением времени и без
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


# feat_CO = ['PT08.S1(CO)','T', 'RH', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'T','PT08.S5(O3)', 'CO(GT)']
feat_CO = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'T' , 'CO(GT)']
# feat_CO = ['PT08.S2(NMHC)', 'T' , 'CO(GT)']
# feat_CO = ['PT08.S1(CO)','T','RH', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)','T', 'CO(GT)']
# feat_CO = ['PT08.S1(CO)', 'CO(GT)']


Degree = 2
# list_delete = []
list_delete = ['T^2','PT08.S2(NMHC)^2']
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

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = T_size)

time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']
X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)

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

poly = PolynomialFeatures(degree=Degree)

print(X_train.head(5))

X_train_poly = pd.DataFrame(poly.fit_transform(X_train))
X_test_poly = pd.DataFrame(poly.fit_transform(X_test))
list_poly =list(poly.get_feature_names_out())
X_train_poly.set_axis(list_poly, axis = 'columns', inplace=True)
X_test_poly.set_axis(list_poly, axis = 'columns', inplace=True)
X_train_poly = X_train_poly.dropna().drop(list_delete, axis=1)
X_test_poly  = X_test_poly.dropna().drop(list_delete, axis=1)

# print(X_train_poly.head(5))
# print(X_test_poly.head(5))

# print(y_train.head(5))
# print(time_train.head(5))                
            
#                                                                       # Lin 

# # machine learning in two lines
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# plotModelResults(lr, X_train=X_train, X_test=X_test, string = "lin",  plot_intervals=True)
# # plotCoefficients(lr, X_train=X_train)
                                                            
                                                                      # Pol
        
lr = LinearRegression()
lr.fit(X_train_poly, y_train)
plotModelResults(lr, X_train=X_train_poly, X_test=X_test_poly, string =  "pol degree = " + str(Degree),  plot_intervals=True)
plotCoefficients(lr, X_train=X_train_poly)

                                                                    # Scaled Model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))
list_X = X_test.columns.values.tolist()
X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

                                                                    # Lin Scaled

# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)

# plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, string = "sc_lin",  plot_intervals=True)
# plotCoefficients(lr, X_train=X_train_scaled)

                                                                    # Pol Scaled

X_train_poly_scaled = pd.DataFrame(poly.fit_transform(X_train_scaled))
X_test_poly_scaled = pd.DataFrame(poly.fit_transform(X_test_scaled))
list_poly_scaled = list(poly.get_feature_names_out())
X_train_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
X_test_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
print(X_train_poly_scaled.columns)

X_train_poly_scaled = X_train_poly_scaled.dropna().drop(list_delete, axis=1)
X_test_poly_scaled  = X_test_poly_scaled.dropna().drop(list_delete, axis=1)
print(X_train_poly_scaled.columns)
# print(X_train_poly_scaled.head(5))
# print(X_test_poly_scaled.head(5))

lr = LinearRegression()
lr.fit(X_train_poly_scaled, y_train)
plotModelResults(lr, X_train=X_train_poly_scaled, X_test=X_test_poly_scaled, string = "pol_sc degree = " + str(Degree),  plot_intervals=True)
plotCoefficients(lr, X_train=X_train_poly_scaled)

                                                                # Model Scaled with hour and weekday


data["hour"] = df_new['datetime'].dt.hour
# data["weekday"] = df_new['datetime'].dt.weekday

y = data.dropna()[feat_target]
X = data.dropna().drop([feat_target], axis=1)
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=T_size)
time_train = X_train.dropna()['datetime']
time_test = X_test.dropna()['datetime']
X_train = X_train.dropna().drop(['datetime'], axis=1)
X_test = X_test.dropna().drop(['datetime'], axis=1)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))
list_X = X_test.columns.values.tolist()
X_train_scaled.set_axis(list_X, axis = 'columns', inplace=True)
X_test_scaled.set_axis(list_X, axis = 'columns', inplace=True)

                                                                # Lin Scaled with hour and weekday

# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)
# plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled,string = "lin_sc with h,d", plot_intervals=True)
# plotCoefficients(lr, X_train=X_train_scaled)

                                                                # Pol Scaled with hour and weekday

# X_train_poly_scaled = pd.DataFrame(poly.fit_transform(X_train_scaled))
# X_test_poly_scaled = pd.DataFrame(poly.fit_transform(X_test_scaled))
# list_poly_scaled = list(poly.get_feature_names_out())
# X_train_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)
# X_test_poly_scaled.set_axis(list_poly_scaled, axis = 'columns', inplace=True)

X_train_poly_scaled['hour'] = X_train_scaled['hour']
# X_train_poly_scaled['weekday'] = X_train_scaled['weekday']

X_test_poly_scaled['hour'] = X_test_scaled['hour']
# X_test_poly_scaled['weekday'] = X_test_scaled['weekday']

lr = LinearRegression()
lr.fit(X_train_poly_scaled, y_train)
plotModelResults(lr, X_train=X_train_poly_scaled, X_test=X_test_poly_scaled, string = "sc_pol with h,d degree = " + str(Degree),  plot_intervals=True)
plotCoefficients(lr, X_train=X_train_poly_scaled)
        
#                                                             # LASSO, RIDGE

from sklearn.linear_model import LassoCV, RidgeCV

                                                     # Lin Ridge Scaled with hour and weekday

# ridge = RidgeCV(cv=tscv)
# ridge.fit(X_train_scaled, y_train)

# plotModelResults(ridge, X_train = X_train_scaled, X_test=X_test_scaled, plot_intervals=True, string ="sc_lin Ridge with h,d")
# plotCoefficients(ridge, X_train = X_train_scaled)

                                                     # Pol Ridge Scaled with hour and weekday

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_poly_scaled, y_train)
plotModelResults(ridge, X_train = X_train_poly_scaled, X_test=X_test_poly_scaled, plot_intervals=True, string ="sc_pol Ridge with h,d" + str(Degree))
plotCoefficients(ridge, X_train = X_train_poly_scaled)

                                                     # Lin LASSO Scaled with hour and weekday


# lasso = LassoCV(cv=tscv)
# lasso.fit(X_train_scaled, y_train)

# plotModelResults(lasso, X_train = X_train_scaled, X_test=X_test_scaled, plot_intervals=True, string ="sc_lin Lasso with h,d")
# plotCoefficients(lasso, X_train = X_train_scaled)

                                                     # Pol LASSO Scaled with hour and weekday


lasso = LassoCV(cv=tscv)
lasso.fit(X_train_poly_scaled, y_train)
plotModelResults(lasso, X_train = X_train_poly_scaled, X_test=X_test_poly_scaled, plot_intervals=True, string ="sc_pol Lasso with h,d" + str(Degree))
plotCoefficients(lasso, X_train = X_train_poly_scaled)


#                                                                 # BOOSTING

from xgboost import XGBRegressor
xgb = XGBRegressor(verbosity=0)


                                                  # Lin XGBR Scaled with hour and weekday

# xgb.fit(X_train_scaled, y_train);
# plotModelResults(xgb,X_train=X_train_scaled,X_test=X_test_scaled, plot_intervals=True, string ="sc_lin XGBR with h,d")

                                                  # Pol XGBR Scaled with hour and weekday

xgb.fit(X_train_poly_scaled, y_train);
plotModelResults(xgb,X_train = X_train_poly_scaled,X_test=X_test_poly_scaled, plot_intervals=True, string ="sc_pol XGBR with h,d")

plt.show()