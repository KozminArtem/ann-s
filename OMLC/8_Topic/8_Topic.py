			

							# https://mlcourse.ai/book/topic08/topic08_sgd_hashing_vowpal_wabbit.html



import warnings
warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

PATH_TO_ALL_DATA = '../Data_OMLC/bank/'
# data_demo = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'weights_heights.csv'))




df = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'bank-full.csv'), sep = ';')
labels = pd.read_csv(os.path.join(PATH_TO_ALL_DATA,'bank.csv'),sep = ';')

print(df.head())

print(labels.head())

df['education'].value_counts().plot.barh()

label_encoder = LabelEncoder()

mapped_education = pd.Series(label_encoder.fit_transform(df['education']))
mapped_education.value_counts().plot.barh()
print(dict(enumerate(label_encoder.classes_)))

df['education'] = mapped_education
print(df.head())


categorical_columns = df.columns[df.dtypes == 'object'].union(['education'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
print(df.head())




def logistic_regression_accuracy_on(dataframe, labels):
    features = dataframe.values()
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    return classification_report(test_labels, logit.predict(test_features))

print(logistic_regression_accuracy_on(df[categorical_columns], labels))
















plt.show()
