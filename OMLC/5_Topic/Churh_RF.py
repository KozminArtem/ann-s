# https://mlcourse.ai/book/topic05/topic5_part2_random_forest.html

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
# %config InlineBackend.figure_format = 'retina'

import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)

# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"

# Load data
df = pd.read_csv(DATA_PATH + "telecom_churn.csv")

# Choose the numeric features
cols = []
for i in df.columns:
    if (df[i].dtype == "float64") or (df[i].dtype == 'int64'):
        cols.append(i)

# Divide the dataset into the input and target
X, y = df[cols].copy(), np.asarray(df["Churn"],dtype='int8')

# # Initialize a stratified split of our dataset for the validation process
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Initialize the classifier with the default parameters
# rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

# # Train it on the training set
# results = cross_val_score(rfc, X, y, cv=skf)

# # Evaluate the accuracy on the test set
# print("CV accuracy score: {:.2f}%".format(results.mean() * 100))


# Initialize the validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Create lists to save the values of accuracy on training and test sets
# train_acc = []
# test_acc = []
# temp_train_acc = []
# temp_test_acc = []
# trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

# for ntrees in trees_grid:
#     rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1)
#     temp_train_acc = []
#     temp_test_acc = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         rfc.fit(X_train, y_train)
#         temp_train_acc.append(rfc.score(X_train, y_train))
#         temp_test_acc.append(rfc.score(X_test, y_test))
#     train_acc.append(temp_train_acc)
#     test_acc.append(temp_test_acc)

# train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
# print("Best CV accuracy is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100,
#                                                         trees_grid[np.argmax(test_acc.mean(axis=1))]))
# plt.style.use('ggplot')

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
# ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
# ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
# ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
# ax.legend(loc='best')
# ax.set_ylim([0.88,1.02])
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("N_estimators");
# plt.show()




# # Create lists to save accuracy values on the training and test sets
# train_acc = []
# test_acc = []
# temp_train_acc = []
# temp_test_acc = []
# max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

# for max_depth in max_depth_grid:
#     rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=max_depth)
#     temp_train_acc = []
#     temp_test_acc = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         rfc.fit(X_train, y_train)
#         temp_train_acc.append(rfc.score(X_train, y_train))
#         temp_test_acc.append(rfc.score(X_test, y_test))
#     train_acc.append(temp_train_acc)
#     test_acc.append(temp_test_acc)

# train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
# print("Best CV accuracy is {:.2f}% with {} max_depth".format(max(test_acc.mean(axis=1))*100,
#                                                         max_depth_grid[np.argmax(test_acc.mean(axis=1))]))

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
# ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
# ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
# ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
# ax.legend(loc='best')
# ax.set_ylim([0.88,1.02])
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Max_depth")
# plt.show()



# # Create lists to save accuracy values on the training and test sets
# train_acc = []
# test_acc = []
# temp_train_acc = []
# temp_test_acc = []
# min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

# for min_samples_leaf in min_samples_leaf_grid:
#     rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
#                                  min_samples_leaf=min_samples_leaf)
#     temp_train_acc = []
#     temp_test_acc = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         rfc.fit(X_train, y_train)
#         temp_train_acc.append(rfc.score(X_train, y_train))
#         temp_test_acc.append(rfc.score(X_test, y_test))
#     train_acc.append(temp_train_acc)
#     test_acc.append(temp_test_acc)

# train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
# print("Best CV accuracy is {:.2f}% with {} min_samples_leaf".format(max(test_acc.mean(axis=1))*100,
#                                                         min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
# ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
# ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
# ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
# ax.legend(loc='best')
# ax.set_ylim([0.88,1.02])
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Min_samples_leaf");
# plt.show()



# # Create lists to save accuracy values on the training and test sets
# train_acc = []
# test_acc = []
# temp_train_acc = []
# temp_test_acc = []
# max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

# for max_features in max_features_grid:
#     rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
#                                  max_features=max_features)
#     temp_train_acc = []
#     temp_test_acc = []
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         rfc.fit(X_train, y_train)
#         temp_train_acc.append(rfc.score(X_train, y_train))
#         temp_test_acc.append(rfc.score(X_test, y_test))
#     train_acc.append(temp_train_acc)
#     test_acc.append(temp_test_acc)

# train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
# print("Best CV accuracy is {:.2f}% with {} max_features".format(max(test_acc.mean(axis=1))*100,
#                                                         max_features_grid[np.argmax(test_acc.mean(axis=1))]))

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
# ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
# ax.fill_between(max_features_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
# ax.fill_between(max_features_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
#                 test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
# ax.legend(loc='best')
# ax.set_ylim([0.88,1.02])
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Max_features");

# plt.show()

# Initialize the set of parameters for exhaustive search and fit
parameters = {'max_features': [4, 7, 10, 13],
              'min_samples_leaf': [1, 3, 5, 7],
              'max_depth': [5, 10, 15, 20]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)

print(gcv.best_params_, gcv.best_score_)

