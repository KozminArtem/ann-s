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
print("GSCV KNN")
df = pd.read_csv('GSCV_KNN_V1.csv') 



list_column = ['param_n_neighbors', 'param_weights','param_p', 'param_leaf_size', 'mean_test_score','std_test_score','rank_test_score']
df_GSCV = df[list_column]
df_GSCV.sort_values(by='rank_test_score', inplace = True)
print(df_GSCV.head(20))
print(df_GSCV.tail(20))


