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






pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print()
print("GSCV ",1000)
df_1000 = pd.read_csv('GSCV_1000_V6_ep.csv') 

list_column = ['param_n_layer', 'param_n_neur','param_activ', 'mean_test_score','std_test_score','rank_test_score']
df_GSCV_1000 = df_1000[list_column]
df_GSCV_1000.sort_values(by='rank_test_score', inplace = True)
print(df_GSCV_1000.head(48))
# print(df_GSCV_1000.tail(20))



print()
print("GSCV ",2000)
df_2000 = pd.read_csv('GSCV_2000_V6_ep.csv') 

list_column = ['param_n_layer', 'param_n_neur','param_activ', 'mean_test_score','std_test_score','rank_test_score']
df_GSCV_2000 = df_2000[list_column]
df_GSCV_2000.sort_values(by='rank_test_score', inplace = True)
print(df_GSCV_2000.head(48))
# print(df_GSCV_2000.tail(20))


















													# VERSION 5


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# print()
# print("GSCV ",500)
# df_500 = pd.read_csv('GSCV_500_V5_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','param_activ', 'mean_test_score','std_test_score','rank_test_score']
# df_GSCV_500 = df_500[list_column]
# df_GSCV_500.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_500.head(90))
# # print(df_GSCV_500.tail(20))



# print()
# print("GSCV ",1000)
# df_1000 = pd.read_csv('GSCV_1000_V5_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','param_activ', 'mean_test_score','std_test_score','rank_test_score']
# df_GSCV_1000 = df_1000[list_column]
# df_GSCV_1000.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_1000.head(90))
# # print(df_GSCV_1000.tail(20))





													# VERSION 4

# print()
# print("GSCV ",100)
# df_100 = pd.read_csv('GSCV_100_V4_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','param_reg', 'mean_test_score','std_test_score','rank_test_score']
# df_GSCV_100 = df_100[list_column]
# df_GSCV_100.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_100.head(20))
# print(df_GSCV_100.tail(20))


# print()
# print("GSCV ",500)
# df_500 = pd.read_csv('GSCV_500_V4_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','param_reg', 'mean_test_score','std_test_score','rank_test_score']
# df_GSCV_500 = df_500[list_column]
# df_GSCV_500.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_500.head(20))
# print(df_GSCV_500.tail(20))




# print()
# print("GSCV ",1000)
# df_1000 = pd.read_csv('GSCV_1000_V4_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','param_reg', 'mean_test_score','std_test_score','rank_test_score']
# df_GSCV_1000 = df_1000[list_column]
# df_GSCV_1000.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_1000.head(20))
# print(df_GSCV_1000.tail(20))





													# VERSION 3

# print()
# print("GSCV ",100)
# df_100 = pd.read_csv('GSCV_100_V3_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_100 = df_100[list_column]
# df_GSCV_100.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_100.head(20))
# print(df_GSCV_100.tail(20))



# print()
# print("GSCV ",500)
# df_500 = pd.read_csv('GSCV_500_V3_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_500 = df_500[list_column]
# df_GSCV_500.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_500.head(20))
# print(df_GSCV_500.tail(20))




# print()
# print("GSCV ",1000)
# df_1000 = pd.read_csv('GSCV_1000_V3_ep.csv') 

# list_column = ['param_l_r', 'param_d_c','param_eps','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_1000 = df_1000[list_column]
# df_GSCV_1000.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_1000.head(20))
# print(df_GSCV_1000.tail(20))




																# VERSION 1
														# l_r opt 0.05 bad 0.005
														# d_c opt 0.001 bad 0.01



# print()
# print("GSCV ",20)
# df_20 = pd.read_csv('GSCV_20ep.csv') 

# list_column = ['param_b_1','param_b_2','param_d_c','param_eps','param_l_r','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_20 = df_20[list_column]
# df_GSCV_20.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_20.head(20))
# print(df_GSCV_20.tail(20))



# print()
# print("GSCV ",100)
# df_100 = pd.read_csv('GSCV_100ep.csv') 

# list_column = ['param_b_1','param_b_2','param_d_c','param_eps','param_l_r','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_500 = df_100[list_column]
# df_GSCV_500.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_500.head(20))
# print(df_GSCV_500.tail(20))



# print()
# print("GSCV ",100)
# df_100 = pd.read_csv('GSCV_100(2)ep.csv') 

# list_column = ['param_b_1','param_b_2','param_d_c','param_eps','param_l_r','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_100 = df_100[list_column]
# df_GSCV_100.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_100.head(20))
# print(df_GSCV_100.tail(20))




# print()
# print("GSCV ",500)
# df_500 = pd.read_csv('GSCV_500ep.csv') 

# list_column = ['param_b_1','param_b_2','param_d_c','param_eps','param_l_r','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_500 = df_500[list_column]
# df_GSCV_500.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_500.head(20))
# print(df_GSCV_500.tail(20))



# print()
# print("GSCV ",1000)
# df_1000 = pd.read_csv('GSCV_1000ep.csv') 

# list_column = ['param_b_1','param_b_2','param_d_c','param_eps','param_l_r','mean_test_score','std_test_score','rank_test_score']
# df_GSCV_1000 = df_1000[list_column]
# df_GSCV_1000.sort_values(by='rank_test_score', inplace = True)
# print(df_GSCV_1000.head(20))
# print(df_GSCV_1000.tail(20))

