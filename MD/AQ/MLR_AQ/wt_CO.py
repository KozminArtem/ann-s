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

df_new['CO(GT)']  = df['CO(GT)']
print(df)
print(df_new)

M = df_new['CO(GT)'].median()
for j in range(len(df_new['CO(GT)'])):
	if df_new['CO(GT)'][j] < -100:
		df_new['CO(GT)'][j] = 0
		# if j > 24 :
		# 	df_new['CO(GT)'][j] = df_new['CO(GT)'][j - 24]	 
		# else:
		# 	df_new['CO(GT)'][j] = df_new['CO(GT)'].median()



for j in range(len(df_new['CO(GT)'])):
	if df_new['CO(GT)'][j] < 0.01:
		df_new['CO(GT)'][j] = 0.5*(df_new['CO(GT)'][j-1] + df_new['CO(GT)'][j+1])
		# if j > 24 :
		# 	df_new['CO(GT)'][j] = df_new['CO(GT)'][j - 24]	 
		# else:
		# 	df_new['CO(GT)'][j] = df_new['CO(GT)'].median()


plt.plot(df_new['datetime'], df_new['CO(GT)'])



sst = np.array(df_new['CO(GT)'])
time = np.arange(0, len(df_new['CO(GT)']), 1)

# time_min = 1048

time_min = 2050
time_max = time_min + 24*7*5




plt.figure()
plt.plot(time[time_min:time_max], sst[time_min:time_max])

import pywt






dt = 1.0


# wavelet = 'mexh'
# min_scale = 0.5
# max_scale = 10



wavelet = 'morl'
min_scale = 0.5
max_scale = 250


# wavelet = 'cmor1-1'
# min_scale = 0.5
# max_scale = 40




scales = np.arange(min_scale, max_scale, 0.1)


[cfs, frequencies] = pywt.cwt(sst[time_min:time_max], scales, wavelet, dt)


period = 1.0/frequencies

print(frequencies)
print(period)

A_scales, B_time = np.meshgrid((time[time_min:time_max]-time_min)/24, period)


plt.figure('pywt: 2D-график для z = w (a,b)')
plt.title('Wavelet Morlet CO(GT)', size=12)
plt.contourf(A_scales, B_time, np.abs(cfs), extended = 'both	')

plt.axhline(y = 24, color = 'green', linestyle = '-')
plt.axhline(y = 24*7, color = 'aqua', linestyle = '-')

# plt.axhline(y = 12, color = 'blue', linestyle = '-.')

ax = plt.gca()
ax.set_ylabel('Period, hour', fontsize = 12)
ax.set_xlabel('Time, day', fontsize = 12)

# plt.axhline(y = 24*7, color = 'green', linestyle = '-')


# plt.savefig('../fig_AQ/Wavelet_7.png')

plt.show()