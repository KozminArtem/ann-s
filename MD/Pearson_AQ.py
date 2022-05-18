# Построение Различных корреляционных матриц от исходных данных, 
# Реализованы фунцкии расстояний Manhattan, Euclidian
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

df_src_AQ = pd.read_csv('AirQualityUCI/AirQualityUCI.csv', sep = ';') 

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows',100000)

# print(df_src_AQ.info())
# print(df_src_AQ.head(-1))

										# Violin Plot
	
feat_float = list(df_src_AQ.columns)
del feat_float[0]
del feat_float[0]
del feat_float[-1]
del feat_float[-1]


for column in feat_float:
    df_src_AQ[column] = [str(x).replace(',','.') for x in df_src_AQ[column]]
    df_src_AQ[column] = df_src_AQ[column].astype(float)

# print(feat_float)

df_src_AQ[feat_float].plot(kind='density', subplots=True, layout=(4, 4), sharex=False, figsize=(16,16))
# # plt.savefig('data_fig/density_raw.png')
# # plt.show()

# _, axes = plt.subplots(2, 2, sharey = True, figsize = (12, 12))
# for value, c in enumerate(feat_float): 
# 	sns.violinplot(x = c, data = df_src_CO, ax = axes[value//2][value % 2]);
# # plt.savefig('data_fig/violin_raw.png')


df_src_diffmean = pd.DataFrame()

df_src_AQ['Date'] = pd.to_datetime(df_src_AQ['Date'], format='%d/%m/%Y')
df_src_AQ['Time'] = pd.to_datetime(df_src_AQ['Time'], format='%H.%M.%S')
df_src_AQ['Date'] = df_src_AQ['Date'].dt.date
df_src_AQ['Time'] = df_src_AQ['Time'].dt.time
df_src_AQ = df_src_AQ.dropna(subset=['Date', 'Time'])
# print(datetime.datetime.combine(df_src_AQ['Date'][0],df_src_AQ['Time'][0]))
df_src_diffmean['datetime'] = pd.Series([datetime.datetime.combine(df_src_AQ['Date'][i], df_src_AQ['Time'][i]) for i in range(df_src_AQ['Date'].size) if df_src_AQ['Time'][i]])

print(df_src_AQ.info())
print(df_src_AQ.head(20))


for collumn in feat_float:
    df_src_diffmean[collumn] = df_src_AQ[collumn]

# for collumn in feat_float:
#     df_src_diffmean[collumn] = df_src_AQ[collumn] - df_src_AQ[collumn].mean()
#     df_src_diffmean[collumn] = df_src_diffmean[collumn]/(df_src_diffmean[collumn].max()-df_src_diffmean[collumn].min())
    


feat_float_diff = list(df_src_diffmean.columns)
del feat_float_diff[0]

x = df_src_diffmean['datetime']

plt.subplots(sharey = True, figsize = (12, 12))

for collumn_diff in feat_float_diff:
    # plt.suptitle(collumn_diff)
    y = df_src_diffmean[collumn_diff]
    plt.plot(x,y,label=str(collumn_diff))


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)   


# # plt.savefig('data_H2SCO_fig/H2SCOT_DiffMean.png')



# # df_src_diffmean[feat_float_diff].plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(12,12))
# # # plt.savefig('data_fig/density_raw.png')
# # # plt.show()

# # _, axes = plt.subplots(2, 2, sharey = True, figsize = (12, 12))
# # for value, c in enumerate(feat_float_diff): 
# #     sns.violinplot(x = c, data = df_src_diffmean, ax = axes[value//2][value % 2]);
# # # plt.savefig('data_fig/violin_raw.png')


# #                                         # Corr


plt.subplots(sharey = True, figsize = (12, 12))
c_m_pear = df_src_diffmean[feat_float_diff].corr(method='pearson')
sns.heatmap(c_m_pear, annot=True)
plt.title("Pearson correlation (norm data)", fontsize=22)
# plt.savefig('data_H2SCO_fig/corr_pear_diff.png')

plt.subplots(sharey = True, figsize = (12, 12))
c_m_kend = df_src_diffmean[feat_float_diff].corr(method='kendall')
sns.heatmap(c_m_kend, annot=True)
plt.title("Kendall correlation (norm data)", fontsize=22)
# plt.savefig('data_H2SCO_fig/corr_kend_diff.png')

plt.subplots(sharey = True, figsize = (12, 12))
c_m_spea = df_src_diffmean[feat_float_diff].corr(method='spearman')
sns.heatmap(c_m_spea, annot=True)
plt.title("Spearman correlation (norm data)", fontsize=22)
# plt.savefig('data_H2SCO_fig/corr_spea_diff.png')

# sns.pairplot(df_src_diffmean[feat_float_diff], kind = 'reg')  
# plt.savefig('data_H2SCO_fig/scat_plot_diff.png')

# def Euclidian_distance(column_1, column_2):
#     norma = max(np.linalg.norm(df_src_diffmean[column_1]), np.linalg.norm(df_src_diffmean[column_2]))
#     if norma > 0.0001:
#         return np.linalg.norm(df_src_diffmean[column_1] - df_src_diffmean[column_2])/norma
#     else: 
#         return 0

# def Manhattan_distance(column_1, column_2): 
#     norma = max(np.linalg.norm(df_src_diffmean[column_1], ord = 1), np.linalg.norm(df_src_diffmean[column_2], ord = 1))
#     # print(norma)
#     if norma > 0.0001:
#         return np.linalg.norm(df_src_diffmean[column_1] - df_src_diffmean[column_2], ord = 1)/norma
#     else: 
#         return 0

# ED = np.empty((12,12), dtype="float32")
# MD = np.empty((12,12), dtype="float32")

# for i, col_1 in enumerate (feat_float_diff):
#     for j, col_2 in enumerate(feat_float_diff):  
#         ED[i][j] = Euclidian_distance(col_1,col_2)
#         MD[i][j] = Manhattan_distance(col_1,col_2)

# plt.subplots(sharey = True, figsize = (12, 10))
# sns.heatmap(ED, yticklabels = feat_float_diff, xticklabels = feat_float_diff, annot=True)
# plt.title("Euclidian distance (norm data)", fontsize=22)
# # plt.savefig('data_H2SCO_fig/dist_eucl_diff.png')

# plt.subplots(sharey = True, figsize = (12, 10))
# sns.heatmap(MD, yticklabels = feat_float_diff, xticklabels = feat_float_diff, annot=True)
# plt.title("Manhattan distance (norm data)", fontsize=22)
# # plt.savefig('data_H2SCO_fig/dist_manh_diff.png')

feat_CO = ['CO(GT)','PT08.S1(CO)','T', 'RH']

for feature in feat_CO:
    df_src_diffmean = df_src_diffmean[df_src_diffmean[feature] > -100]

x = df_src_diffmean['datetime']

plt.subplots(sharey = True, figsize = (12, 12))

for collumn_diff in feat_CO:
    # plt.suptitle(collumn_diff)
    y = df_src_diffmean[collumn_diff]
    plt.plot(x,y,label=str(collumn_diff))

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)   

plt.show()