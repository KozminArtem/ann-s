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


df_src_CO = pd.read_csv('220131_H2S_high_T/field_src+CO/G2000315.csv') 


df_src_CO = df_src_CO.iloc[2000:20000]

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows',100000)


print(df_src_CO.info())
print(df_src_CO.head(20))





										# Violin Plot
	

feat_float = list(df_src_CO.columns)
del feat_float[0]

# df_src_CO[feat_float].plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(12,12))
# # plt.savefig('data_fig/density_raw.png')
# # plt.show()

# _, axes = plt.subplots(2, 2, sharey = True, figsize = (12, 12))
# for value, c in enumerate(feat_float): 
# 	sns.violinplot(x = c, data = df_src_CO, ax = axes[value//2][value % 2]);
# # plt.savefig('data_fig/violin_raw.png')


# # # 										# Corr


# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_pear = df_src_CO[feat_float].corr(method='pearson')
# sns.heatmap(c_m_pear)
# # plt.savefig('data_fig/corr_pear_src.png')

# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_kend = df_src_CO[feat_float].corr(method='kendall')
# sns.heatmap(c_m_kend)
# # plt.savefig('data_fig/corr_kend_src.png')

# plt.subplots(sharey = True, figsize = (12, 10))
# c_m_spea = df_src_CO[feat_float].corr(method='spearman')
# sns.heatmap(c_m_spea)
# # plt.savefig('data_fig/corr_spea_src.png')


									# Scatterplot



						# Scatterplot matrix - Матрица диаграммы рассеяния

# sns.pairplot(df_src_CO[feat_float])
# plt.savefig('data_fig/corr_Scatterplot_raw.png')






df_src_CO['datetime'] = df_src_CO['date'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

# x = df_src_CO['datetime']

# for collumn in feat_float:
# 	plt.subplots(sharey = True, figsize = (12, 10))
# 	plt.suptitle(collumn)
# 	y = df_src_CO[collumn]
# 	plt.plot(x,y)


df_src_diffmean = pd.DataFrame()

df_src_diffmean['datetime'] =  df_src_CO['datetime']  


for collumn in feat_float:
    df_src_diffmean[collumn] = df_src_CO[collumn] - df_src_CO[collumn].mean()
    df_src_diffmean[collumn] = df_src_diffmean[collumn]/(df_src_diffmean[collumn].max()-df_src_diffmean[collumn].min())
    
df_src_diffmean['H2Sop1'] = -1*df_src_diffmean['H2Sop1']*1.5
# df_src_diffmean['H2S_summ'] = df_src_diffmean['H2Sop1'] + df_src_diffmean['H2Sop2']








feat_float_diff = list(df_src_diffmean.columns)
del feat_float_diff[0]

x = df_src_diffmean['datetime']

plt.subplots(sharey = True, figsize = (12, 10))

for collumn_diff in feat_float_diff:
    # plt.suptitle(collumn_diff)
    y = df_src_diffmean[collumn_diff]
    plt.plot(x,y,label=str(collumn_diff))


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)   


# plt.savefig('data_H2SCO_fig/H2SCOT_DiffMean.png')



# df_src_diffmean[feat_float_diff].plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(12,12))
# # plt.savefig('data_fig/density_raw.png')
# # plt.show()

# _, axes = plt.subplots(2, 2, sharey = True, figsize = (12, 12))
# for value, c in enumerate(feat_float_diff): 
#     sns.violinplot(x = c, data = df_src_diffmean, ax = axes[value//2][value % 2]);
# # plt.savefig('data_fig/violin_raw.png')


# #                                         # Corr


plt.subplots(sharey = True, figsize = (12, 10))
c_m_pear = df_src_diffmean[feat_float_diff].corr(method='pearson')
sns.heatmap(c_m_pear, annot=True)
plt.title("Pearson correlation (norm data)", fontsize=22)
plt.savefig('data_H2SCO_fig/corr_pear_diff.png')

plt.subplots(sharey = True, figsize = (12, 10))
c_m_kend = df_src_diffmean[feat_float_diff].corr(method='kendall')
sns.heatmap(c_m_kend, annot=True)
plt.title("Kendall correlation (norm data)", fontsize=22)
plt.savefig('data_H2SCO_fig/corr_kend_diff.png')

plt.subplots(sharey = True, figsize = (12, 10))
c_m_spea = df_src_diffmean[feat_float_diff].corr(method='spearman')
sns.heatmap(c_m_spea, annot=True)
plt.title("Spearman correlation (norm data)", fontsize=22)
plt.savefig('data_H2SCO_fig/corr_spea_diff.png')



sns.pairplot(df_src_diffmean[feat_float_diff], kind = 'reg')
plt.savefig('data_H2SCO_fig/scat_plot_diff.png')




def Euclidian_distance(column_1, column_2):
    norma = max(np.linalg.norm(df_src_diffmean[column_1]), np.linalg.norm(df_src_diffmean[column_2]))
    if norma > 0.0001:
        return np.linalg.norm(df_src_diffmean[column_1] - df_src_diffmean[column_2])/norma
    else: 
        return 0

def Manhattan_distance(column_1, column_2):
 
    norma = max(np.linalg.norm(df_src_diffmean[column_1], ord = 1), np.linalg.norm(df_src_diffmean[column_2], ord = 1))
    # print(norma)
    if norma > 0.0001:
        return np.linalg.norm(df_src_diffmean[column_1] - df_src_diffmean[column_2], ord = 1)/norma
    else: 
        return 0



ED = np.empty((4,4), dtype="float32")
MD = np.empty((4,4), dtype="float32")

for i, col_1 in enumerate (feat_float_diff):
    for j, col_2 in enumerate(feat_float_diff):  
        ED[i][j] = Euclidian_distance(col_1,col_2)
        MD[i][j] = Manhattan_distance(col_1,col_2)



plt.subplots(sharey = True, figsize = (12, 10))
sns.heatmap(ED, yticklabels = feat_float_diff, xticklabels = feat_float_diff, annot=True)
plt.title("Euclidian distance (norm data)", fontsize=22)
plt.savefig('data_H2SCO_fig/dist_eucl_diff.png')

plt.subplots(sharey = True, figsize = (12, 10))
sns.heatmap(MD, yticklabels = feat_float_diff, xticklabels = feat_float_diff, annot=True)
plt.title("Manhattan distance (norm data)", fontsize=22)
plt.savefig('data_H2SCO_fig/dist_manh_diff.png')

plt.show()