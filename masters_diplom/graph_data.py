# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8 
import datetime


# df_2 = pd.read_csv('rooftop_raw/G10002F1.csv')
# df_3 = pd.read_csv('rooftop_raw/G1000317.csv')
# df_NO2 = pd.read_csv('rooftop_raw/Optec_NO2.csv')
# df_O3 = pd.read_csv('rooftop_raw/Optec_O3.csv')

# data_6 = pd.read_csv('rooftop_raw/ROOFTOP_G10002A3.csv')
# data_7 = pd.read_csv('rooftop_raw/ROOFTOP_G10002F1.csv')
# data_8 = pd.read_csv('rooftop_raw/ROOFTOP_G1000317.csv')
# print(data_6.info())
# print(data_6.head(5))
# print(data_7.info())
# print(data_7.head(5))
# print(data_8.info())
# print(data_8.head(5))
						
										# G10002A3

df_1 = pd.read_csv('rooftop_raw/G10002A3.csv')										

# print(df_1.info())
# print(df_1.head(5))

# #	 Column   Non-Null Count  Dtype  
# ---  ------   --------------  -----  
#  0   date     39284 non-null  object 
#  1   NO2op1   39284 non-null  float64
#  2   NO2op2   39284 non-null  float64
#  3   NO2t     39284 non-null  float64
#  4   O3op1    39284 non-null  float64
#  5   O3op2    39284 non-null  float64
#  6   O3t      39284 non-null  float64
#  7   COop1    39284 non-null  float64
#  8   COop2    39284 non-null  float64
#  9   COt      39284 non-null  float64
			# Пустые
#  10  NO2_RAW  39284 non-null  int64  
#  11  O3_RAW   39284 non-null  int64  
#  12  CO_RAW   39284 non-null  int64  

# print(df_1.columns)
# temp = ['date','NO2_RAW','O3_RAW','CO_RAW']

feat_float = ['NO2op1','NO2op2','NO2t', 'O3op1', 'O3op2', 'O3t', 'COop1','COop2','COt']

# df_1[feat_float].plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(10, 8))
# plt.savefig('data_fig/density_G10002A3.png')

# _, axes = plt.subplots(3, 3, sharey = True, figsize = (12, 12))
# for value, c in enumerate(feat_float): 
# 	sns.violinplot(x = c, data = df_1, ax = axes[value//3][value % 3]);
# plt.savefig('data_fig/violin_G10002A3.png')

# axes = plt.subplots(1, 1, sharey = True, figsize = (8, 8))
# sns.violinplot(x = 'NO2t', data = df_1, dodge = True, color = "g");
# sns.violinplot(x = 'O3t', data = df_1,dodge = True, color = "b");
# sns.violinplot(x = 'COt', data = df_1,dodge = True, color = "r");
# plt.savefig('data_fig/violin_t_G10002A3.png')



# c_m_pear = df_1[feat_float].corr(method='pearson')
# sns.heatmap(c_m_pear)
# plt.savefig('data_fig/corr_pearson_G10002A3.png')

# c_m_kend = df_1[feat_float].corr(method='kendall')
# sns.heatmap(c_m_kend)
# plt.savefig('data_fig/corr_kendall_G10002A3.png')

# c_m_spea = df_1[feat_float].corr(method='spearman')
# sns.heatmap(c_m_spea)
# plt.savefig('data_fig/corr_spearman_G10002A3.png')


# sns.jointplot(x = 'NO2op1', y = 'NO2op2', data = df_1, kind = 'scatter');
# sns.jointplot('NO2op1', 'NO2op2', data = df_1, kind = "kde", color = "g");
# Scatterplot matrix - Матрица диаграммы рассеяния
# sns.pairplot(df_1[feat_float])
# plt.savefig('data_fig/corr_Scatterplot_G10002A3.png')

# print(df_1.info())
# print(df_1.head(5))
# print(df_1['date'])

df_1['datetime'] = df_1['date'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

x = df_1['datetime']
y = df_1['NO2op1']

# plot
plt.plot(x,y)







# print(df_2.info())
# print(df_2.head(5))

# print(df_3.info())
# print(df_3.head(5))

# print(df_NO2.info())
# print(df_NO2.head(5))

# print(df_O3.info())
# print(df_O3.head(5))







plt.show()

