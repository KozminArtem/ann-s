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

								
								# Знакомство с файлами


df_raw = pd.read_csv('220131_H2S_high_T/field_raw/G10002A3.csv')										
df_src = pd.read_csv('220131_H2S_high_T/field_src/G10002A3.csv')	
df_src_CO = pd.read_csv('220131_H2S_high_T/field_src+CO/G2000301.csv') 
df_src_G1G2 = pd.read_csv('220131_H2S_high_T/field_src_G1+G2/G2000301.csv') 
df_src_meteo = pd.read_csv('220131_H2S_high_T/field_src+meteo/G2000309.csv')


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# print(df_raw.info())
# print(df_raw.head(5))
# print(df_raw.columns)

 # 0   date    31150 non-null  object 
 # 1   COop1   30977 non-null  float64
 # 2   COop2   31150 non-null  float64
 # 3   COt     13410 non-null  float64
 # 4   NO2op1  31150 non-null  float64
 # 5   NO2op2  31150 non-null  float64
 # 6   NO2t    13659 non-null  float64
 # 7   O3op1   31150 non-null  float64
 # 8   O3op2   31150 non-null  float64
 # 9   O3t     13659 non-null  float64
 # 10  CO      30977 non-null  float64
 # 11  NO2     30977 non-null  float64
 # 12  O3      30977 non-null  float64
 # 13  T       17495 non-null  float64
 # dtypes: float64(13), object(1)




print(df_src.info())
print(df_src.head(5))	
print(df_src.columns)

#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   date    31136 non-null  object 
#  1   COop1   30885 non-null  float64
#  2   COop2   31136 non-null  float64
#  3   NO2op1  31136 non-null  float64
#  4   NO2op2  31136 non-null  float64
#  5   O3op1   31136 non-null  float64
#  6   O3op2   31136 non-null  float64
#  7   T       31136 non-null  float64

# print(df_src_CO.info())
# print(df_src_CO.head(5))	
# print(df_src_CO.columns)

# #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   date    31136 non-null  object 
#  1   COop1   30885 non-null  float64
#  2   COop2   31136 non-null  float64
#  3   NO2op1  31136 non-null  float64
#  4   NO2op2  31136 non-null  float64
#  5   O3op1   31136 non-null  float64
#  6   O3op2   31136 non-null  float64
#  7   T       31136 non-null  float64
# dtypes: float64(7), object(1)


# print(df_src_G1G2.info())
# print(df_src_G1G2.head(5))	
# print(df_src_G1G2.columns)

#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   date    27702 non-null  object 
#  1   COop1   27702 non-null  float64
#  2   COop2   27702 non-null  float64
#  3   NO2op1  27702 non-null  float64
#  4   NO2op2  27702 non-null  float64
#  5   O3op1   27702 non-null  float64
#  6   O3op2   27702 non-null  float64
#  7   H2Sop1  27702 non-null  float64
#  8   H2Sop2  27702 non-null  float64
#  9   SO2op1  27702 non-null  float64
#  10  SO2op2  27702 non-null  float64
#  11  RH      27702 non-null  float64
#  12  T       27702 non-null  float64
# dtypes: float64(12), object(1)


# print(df_src_meteo.info())
# print(df_src_meteo.head(5))	
# print(df_src_meteo.columns)

#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   date    24329 non-null  object 
#  1   H2Sop1  24329 non-null  float64
#  2   H2Sop2  24329 non-null  float64
#  3   SO2op1  24329 non-null  float64
#  4   SO2op2  24329 non-null  float64
#  5   RH      24329 non-null  float64
#  6   CO      24329 non-null  float64
#  7   NO2     24329 non-null  float64
#  8   T       24329 non-null  float64
# dtypes: float64(8), object(1)	
								


										# Violin Plot
	

# feat_float = list(df_raw.columns)
# del feat_float[0]
# del feat_float[-1]

feat_float_src = list(df_src.columns)
del feat_float_src[0]





# df_raw[feat_float].plot(kind='density', subplots=True, layout=(4, 3), sharex=False, figsize=(12,10))
# plt.savefig('data_fig/density_raw.png')
# # plt.show()

# _, axes = plt.subplots(4, 3, sharey = True, figsize = (15, 15))
# for value, c in enumerate(feat_float): 
# 	sns.violinplot(x = c, data = df_raw, ax = axes[value//3][value % 3]);
# plt.savefig('data_fig/violin_raw.png')



df_src[feat_float_src].plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(12,12))
plt.savefig('data_fig/density_src.png')
# plt.show()

_, axes = plt.subplots(3, 3, sharey = True, figsize = (12, 12))
for value, c in enumerate(feat_float_src): 
	sns.violinplot(x = c, data = df_src, ax = axes[value//3][value % 3]);
plt.savefig('data_fig/violin_src.png')



										# Corr


# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_pear = df_raw[feat_float].corr(method='pearson')
# sns.heatmap(c_m_pear)
# plt.savefig('data_fig/corr_pear_raw.png')

# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_kend = df_raw[feat_float].corr(method='kendall')
# sns.heatmap(c_m_kend)
# plt.savefig('data_fig/corr_kend_raw.png')

# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_spea = df_raw[feat_float].corr(method='spearman')
# sns.heatmap(c_m_spea)
# plt.savefig('data_fig/corr_spea_raw.png')



plt.subplots(sharey = True, figsize = (12, 10))
c_m_pear = df_src[feat_float_src].corr(method='pearson')
sns.heatmap(c_m_pear)
plt.savefig('data_fig/corr_pear_src.png')

plt.subplots(sharey = True, figsize = (12, 10))
c_m_kend = df_src[feat_float_src].corr(method='kendall')
sns.heatmap(c_m_kend)
plt.savefig('data_fig/corr_kend_src.png')

plt.subplots(sharey = True, figsize = (12, 10))
c_m_spea = df_src[feat_float_src].corr(method='spearman')
sns.heatmap(c_m_spea)
plt.savefig('data_fig/corr_spea_src.png')





									# Scatterplot


# sns.jointplot(x = 'NO2op1', y = 'NO2op2', data = df_raw, kind = 'scatter');
# sns.jointplot('NO2op1', 'NO2op2', data = df_raw, kind = "kde", color = "g");

						# Scatterplot matrix - Матрица диаграммы рассеяния

# sns.pairplot(df_raw[feat_float])
# plt.savefig('data_fig/corr_Scatterplot_raw.png')

sns.pairplot(df_src[feat_float_src])
plt.savefig('data_fig/corr_Scatterplot_src.png')




# df_raw['datetime'] = df_raw['date'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

# x = df_raw['datetime']
# for collumn in feat_float:
# 	plt.subplots(sharey = True, figsize = (12, 10))
# 	plt.suptitle(collumn)
# 	y = df_raw[collumn]
# 	plt.plot(x,y)

df_src['datetime'] = df_src['date'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
x = df_src['datetime']
for collumn in feat_float_src:
	plt.subplots(sharey = True, figsize = (12, 10))
	plt.suptitle(collumn)
	y = df_src[collumn]
	plt.plot(x,y)






plt.show()