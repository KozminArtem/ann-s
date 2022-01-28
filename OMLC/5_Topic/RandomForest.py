# https://mlcourse.ai/book/topic05/topic5_part2_random_forest.html

# Disable warnings in Anaconda
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

# n_train = 150  
# n_test = 1000  
# noise = 0.1

# # Generate data
# def f(x):
#     x = x.ravel()
#     return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

# def generate(n_samples, noise):
#     X = np.random.rand(n_samples) * 10 - 5
#     X = np.sort(X).ravel()
#     y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2)\
#         + np.random.normal(0.0, noise, n_samples)
#     X = X.reshape((n_samples, 1))

#     return X, y

# X_train, y_train = generate(n_samples=n_train, noise=noise)
# X_test, y_test = generate(n_samples=n_test, noise=noise)

# # One decision tree regressor
# dtree = DecisionTreeRegressor().fit(X_train, y_train)
# d_predict = dtree.predict(X_test)

# plt.figure(figsize=(10, 6))
# plt.plot(X_test, f(X_test), "b")
# plt.scatter(X_train, y_train, c="b", s=20)
# plt.plot(X_test, d_predict, "g", lw=2)
# plt.xlim([-5, 5])
# plt.title("Decision tree, MSE = %.2f"
#           % np.sum((y_test - d_predict) ** 2))

# # Bagging with a decision tree regressor
# bdt = BaggingRegressor(DecisionTreeRegressor()).fit(X_train, y_train)
# bdt_predict = bdt.predict(X_test)

# plt.figure(figsize=(10, 6))
# plt.plot(X_test, f(X_test), "b")
# plt.scatter(X_train, y_train, c="b", s=20)
# plt.plot(X_test, bdt_predict, "y", lw=2)
# plt.xlim([-5, 5])
# plt.title("Bagging for decision trees, MSE = %.2f" % np.sum((y_test - bdt_predict) ** 2));

# # Random Forest
# rf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
# rf_predict = rf.predict(X_test)

# plt.figure(figsize=(10, 6))
# plt.plot(X_test, f(X_test), "b")
# plt.scatter(X_train, y_train, c="b", s=20)
# plt.plot(X_test, rf_predict, "r", lw=2)
# plt.xlim([-5, 5])
# plt.title("Random forest, MSE = %.2f" % np.sum((y_test - rf_predict) ** 2));
# plt.show()

np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
X_train_circles, X_test_circles, y_train_circles, y_test_circles = \
    train_test_split(X, y, test_size=0.2)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_circles, y_train_circles)

plt.figure(figsize=(8, 4))
x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Decision tree")

b_dtree = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=300, random_state=42)
b_dtree.fit(X_train_circles, y_train_circles)

plt.figure(figsize=(8, 4))
x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = b_dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Bagging (decision trees)")

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_circles, y_train_circles)

plt.figure(figsize=(8, 4))
x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = rf.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Random forest")
plt.show()


