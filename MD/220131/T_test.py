import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings 
warnings.filterwarnings(action='once')

from pylab import rcParams
import datetime

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0


df_src = pd.read_csv('220131_H2S_high_T/field_src/G10002A3.csv')	

# df_src = pd.read_csv('220131_H2S_high_T/field_src/G2000301.csv')    




pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows',10)

print(df_src.info())


feat_float_src = list(df_src.columns)
del feat_float_src[0]
del feat_float_src[-1]

df_src['datetime'] = df_src['date'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

x = df_src['datetime']


                    # Scatteplot

colors = [plt.cm.tab10(i/float(len(feat_float_src)-1)) for i in range(len(feat_float_src))]


for i, column in enumerate(feat_float_src):
    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.scatter('T', column, data = df_src, s=20, c = colors[i], label=str(column))
    plt.title("Scatterplot: " + str(column) + "(T)", fontsize=22)
    plt.savefig('data_T_fig/scatter_' + str(column) + '.png')


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)    
plt.show()  
