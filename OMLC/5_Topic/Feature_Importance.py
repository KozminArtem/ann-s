# https://mlcourse.ai/book/topic05/topic5_part3_feature_importance.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
data = iris["data"]
target = iris["target"]

data = pd.DataFrame(data, columns=iris["feature_names"])
print(data.head())
target = pd.Series(target).map({0: 0, 1: 0, 2: 1})

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=17)
rfc.fit(data, target)
tree_list = rfc.estimators_

from sklearn import tree

# plt.figure(figsize=(10, 6))
# tree.plot_tree(
#     tree_list[0],
#     filled=True,
#     feature_names=iris["feature_names"],
#     class_names=["Y", "N"],
#     node_ids=True,
# )
# plt.show()

# plt.figure(figsize=(16, 12))
# tree.plot_tree(
#     tree_list[1],
#     filled=True,
#     feature_names=iris["feature_names"],
#     class_names=["Y", "N"],
#     node_ids=True,
# )
# plt.show()

# plt.figure(figsize=(6, 4))
# tree.plot_tree(
#     tree_list[2],
#     filled=True,
#     feature_names=iris["feature_names"],
#     class_names=["Y", "N"],
#     node_ids=True,
# )
# plt.show()

print(iris["feature_names"])
print(rfc.feature_importances_)
