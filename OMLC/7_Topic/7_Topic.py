        

                                        # https://mlcourse.ai/book/topic07/topic7_pca_clustering.html#topic07


                        # Topic 7. Unsupervised learning: PCA and clustering

                        # Principal Component Analysis (PCA)



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="white")
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition

# # Loading the dataset
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # Let's create a beautiful 3d-plot
# fig = plt.figure(1, figsize=(6, 5))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

# plt.cla()

# for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
#     ax.text3D(
#         X[y == label, 0].mean(),
#         X[y == label, 1].mean() + 1.5,
#         X[y == label, 2].mean(),
#         name,
#         horizontalalignment="center",
#         bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
#     )
# # Change the order of labels, so that they match
# y_clr = np.choose(y, [1, 2, 0]).astype(np.float)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=plt.cm.nipy_spectral)

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])



# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# # Train, test splits
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42
# )

# # Decision trees with depth = 2
# clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# clf.fit(X_train, y_train)
# preds = clf.predict_proba(X_test)
# print("Accuracy: {:.5f}".format(accuracy_score(y_test, preds.argmax(axis=1))))




# plt.subplots(sharey = True, figsize = (12, 12))

# # Using PCA from sklearn PCA
# pca = decomposition.PCA(n_components=2)
# X_centered = X - X.mean(axis=0)
# pca.fit(X_centered)
# X_pca = pca.transform(X_centered)

# # Plotting the results of PCA
# plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], "bo", label="Setosa")
# plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], "go", label="Versicolour")
# plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], "ro", label="Virginica")
# plt.legend(loc=0);


# # Test-train split and apply PCA
# X_train, X_test, y_train, y_test = train_test_split(
#     X_pca, y, test_size=0.3, stratify=y, random_state=42
# )

# clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# clf.fit(X_train, y_train)
# preds = clf.predict_proba(X_test)
# print("Accuracy: {:.5f}".format(accuracy_score(y_test, preds.argmax(axis=1))))


# for i, component in enumerate(pca.components_):
#     print(
#         "{} component: {}% of initial variance".format(
#             i + 1, round(100 * pca.explained_variance_ratio_[i], 2)
#         )
#     )
#     print(
#         " + ".join(
#             "%.3f x %s" % (value, name)
#             for value, name in zip(component, iris.feature_names)
#         )
#     )

# digits = datasets.load_digits()
# X = digits.data
# y = digits.target

# # f, axes = plt.subplots(5, 2, sharey=True, figsize=(16,6))
# plt.figure(figsize=(16, 6))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(X[i, :].reshape([8, 8]), cmap="gray");


# pca = decomposition.PCA(n_components=2)
# X_reduced = pca.fit_transform(X)

# print("Projecting %d-dimensional data to 2D" % X.shape[1])

# plt.figure(figsize=(12, 10))
# plt.scatter(
#     X_reduced[:, 0],
#     X_reduced[:, 1],
#     c=y,
#     edgecolor="none",
#     alpha=0.7,
#     s=40,
#     cmap=plt.cm.get_cmap("nipy_spectral", 10),
# )
# plt.colorbar()
# plt.title("MNIST. PCA projection");


# # pca = decomposition.PCA(n_components=3)
# # X_reduced = pca.fit_transform(X)
# # print("Projecting %d-dimensional data to 2D" % X.shape[1])
# # fig = plt.figure(1, figsize=(6, 5))
# # plt.clf()
# # ax = Axes3D(fig, elev=48, azim=134)
# # plt.cla()
# # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.nipy_spectral)






# pca = decomposition.PCA().fit(X)

# plt.figure(figsize=(10, 7))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), color="k", lw=2)
# plt.xlabel("Number of components")
# plt.ylabel("Total explained variance")
# plt.xlim(0, 63)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.axvline(21, c="b")
# plt.axhline(0.9, c="r")





                                            # Clustering



# Let's begin by allocation 3 cluster's points
X = np.zeros((150, 2))

np.random.seed(seed=42)
X[:50, 0] = np.random.normal(loc=0.0, scale=0.3, size=50)
X[:50, 1] = np.random.normal(loc=0.0, scale=0.3, size=50)

X[50:100, 0] = np.random.normal(loc=2.0, scale=0.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=0.2, size=50)

X[100:150, 0] = np.random.normal(loc=-1.0, scale=0.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=0.5, size=50)

plt.figure(figsize=(5, 5))
plt.plot(X[:, 0], X[:, 1], "bo");



# Scipy has function that takes 2 tuples and return
# calculated distance between them
from scipy.spatial.distance import cdist

# Randomly allocate the 3 centroids
np.random.seed(seed=42)
centroids = np.random.normal(loc=0.0, scale=1.0, size=6)
centroids = centroids.reshape((3, 2))

cent_history = []
cent_history.append(centroids)

for i in range(3):
    # Calculating the distance from a point to a centroid
    distances = cdist(X, centroids)
    # Checking what's the closest centroid for the point
    labels = distances.argmin(axis=1)

    # Labeling the point according the point's distance
    centroids = centroids.copy()
    centroids[0, :] = np.mean(X[labels == 0, :], axis=0)
    centroids[1, :] = np.mean(X[labels == 1, :], axis=0)
    centroids[2, :] = np.mean(X[labels == 2, :], axis=0)

    cent_history.append(centroids)





# Let's plot K-means
plt.figure(figsize=(8, 8))
for i in range(4):
    distances = cdist(X, cent_history[i])
    labels = distances.argmin(axis=1)

    plt.subplot(2, 2, i + 1)
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], "bo", label="cluster #1")
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], "co", label="cluster #2")
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], "mo", label="cluster #3")
    plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], "rX")
    plt.legend(loc=0)
    plt.title("Step {:}".format(i + 1))




from sklearn.cluster import KMeans


plt.figure(figsize=(8, 8))


inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))


plt.plot(range(1, 8), inertia, marker="s")
plt.xlabel("$k$")
plt.ylabel("$J(C_k)$");






plt.show()
