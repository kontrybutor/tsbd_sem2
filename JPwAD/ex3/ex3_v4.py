
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.cluster import KMeans


def warn(*args, **kwargs):
    pass


warnings.warn = warn

filename = 'spambase.data'
spam_df = pd.read_csv(filename, sep=',', header=None)
print(spam_df.head())
print(spam_df.describe())

data = pd.DataFrame(spam_df).values.astype('float64')
X = data[:, :-1]  # all rows, no label
y = data[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
scores = []
losses = []
start1 = timer()
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter,)
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X, y, cv=5)
    # scores.append(model.score(X_test, y_test))
    scores.append(cv_scores.mean())
    losses.append(model.loss_)
end1 = timer()
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate before PCA")
plt.ylim([0.0, 1.0])
plt.plot(n_iters, losses)
plt.show()
plt.title("Effect of n_iter for MLP before PCA")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, scores)
plt.show()
# print("MLP Accuracy on input dataset", model.score(X_test, y_test))
print("MLP Accuracy on input dataset: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores)))
print("Elapsed time:", end1 - start1)

pca = PCA(n_components=2)
pca.fit(X_scaled)
pca_df = pca.fit_transform(X_scaled)

pca_data = pd.DataFrame(pca_df)
pca_data[2] = spam_df[[57]]
pca_data_val = pca_data.values.astype('float64')
# print(pca_data.head())
X_pca = pca_data_val[:, :-1]  # all rows, no label
y_pca = pca_data_val[:, -1]

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.3, random_state=0)
n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
pca_scores = []
losses = []
start2 = timer()
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter, )
    model.fit(X_train_pca, y_train_pca)
    cv_scores = cross_val_score(model, X_pca, y_pca, cv=5)
    # pca_scores.append(model.score(X_test_pca, y_test_pca))
    pca_scores.append(cv_scores.mean())
    losses.append(model.loss_)
end2 = timer()
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate after PCA")
plt.ylim([0.0, 1.0])
plt.plot(n_iters, losses)
plt.show()
plt.title("Effect of n_iter for MLP after PCA")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, pca_scores)
plt.show()
# print("MLP Accuracy after PCA:", model.score(X_test_pca, y_test_pca))
print("MLP Accuracy after PCA: %0.2f (+/- %0.2f)" % (np.mean(pca_scores), np.std(pca_scores)))
print("Elapsed time:", end2 - start2)


# print(spam_df.var().sort_values())
spam_df_with_3_attrs = spam_df[[46, 50, 57]]
data_var = spam_df_with_3_attrs.values.astype('float64')
X_var = data_var[:, :-1]  # all rows, no label
y_var = data_var[:, -1]

X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(X_var, y_var, test_size=0.3, random_state=0)

n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
var_scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter, )
    model.fit(X_train_var, y_train_var)
    cv_scores = cross_val_score(model, X_var, y_var, cv=5)
    # var_scores.append(model.score(X_test_var, y_test_var))
    var_scores.append(cv_scores.mean())
    losses.append(model.loss_)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate with 2 attrs with smallest variance")
plt.ylim([0.0, 1.0])
plt.plot(n_iters, losses)
plt.show()
plt.title("Effect of n_iter for MLP with 2 attrs with smallest variance")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, var_scores)
plt.show()
# print("MLP Accuracy with 2 attrs with smallest variance:", model.score(X_test_var, y_test_var))
print("MLP Accuracy for attrs with smallest variance: %0.2f (+/- %0.2f)" % (np.mean(var_scores), np.std(var_scores)))

chi2_features = SelectKBest(chi2, k=2)
X_kbest_features = chi2_features.fit_transform(X, y)

print('Original feature number:', X.shape[1])
print('Reduced feature number:', X_kbest_features.shape[1])
X_kbest_features = pd.DataFrame(X_kbest_features)
X_kbest_features[2] = spam_df[[57]]
chi_data = X_kbest_features.values
X_chi = chi_data[:, :-1]  # all rows, no label
y_chi = chi_data[:, -1]

X_train_chi, X_test_chi, y_train_chi, y_test_chi = train_test_split(X_chi, y_chi, test_size=0.3, random_state=0)

n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
chi_scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=n_iter, )
    model.fit(X_train_chi, y_train_chi)
    cv_scores = cross_val_score(model, X_chi, y_chi, cv=5)
    # chi_scores.append(model.score(X_test_chi, y_test_chi))
    chi_scores.append(cv_scores.mean())
    losses.append(model.loss_)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate after chi2 test")
plt.ylim([0.0, 1.0])
plt.plot(n_iters, losses)
plt.show()
plt.title("Effect of n_iter for MLP after chi2 test")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, chi_scores)
plt.show()
# print("MLP Accuracy after chi2 test:", model.score(X_test_chi, y_test_chi))
print("MLP Accuracy for attrs from chi2 test: %0.2f (+/- %0.2f)" % (np.mean(chi_scores), np.std(chi_scores)))

plt.title("Summary")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, scores, 'r', label='input')
plt.plot(n_iters, pca_scores, 'g', label='PCA')
plt.plot(n_iters, var_scores, 'b', label='smallest variance')
plt.plot(n_iters, chi_scores, 'y', label='chi2_test')
plt.legend()
plt.show()

################################# CLUSTERING ################################

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for initial dataset')
plt.show()

input_kmeans = KMeans(n_clusters=5)
input_kmeans.fit(X)
labels = input_kmeans.predict(X)
centroids = input_kmeans.cluster_centers_
print("Sum of squared distances of samples to their closest cluster center for initial dataset:", input_kmeans.inertia_)

#######################################################################

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_pca)
    distortions.append(sum(np.min(cdist(X_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for PCA')
plt.show()

pca_kmeans = KMeans(n_clusters=2)
pca_kmeans.fit(X_pca)
labels = pca_kmeans.predict(X_pca)
centroids = pca_kmeans.cluster_centers_
print("Sum of squared distances of samples to their closest cluster center for PCA :", pca_kmeans.inertia_)
plt.figure(figsize=(10, 10))
plt.subplot(211)
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'black'}
colors = map(lambda x: colmap[x+1], y)
plt.scatter(pca_data[0], pca_data[1], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
# plt.xlim(0, 2000)
# plt.ylim(0, 2000)
plt.title('Original clusters for PCA')

plt.subplot(212)
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'black'}
colors = map(lambda x: colmap[x+1], labels)
plt.scatter(pca_data[0], pca_data[1], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
# plt.xlim(0, 2000)
# plt.ylim(0, 2000)
plt.title('Clusters after PCA')
plt.show()

#######################################################################

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_var)
    distortions.append(sum(np.min(cdist(X_var, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for attrs with smallest variance')
plt.show()

var_kmeans = KMeans(n_clusters=2)
var_kmeans.fit(X_var)
labels = var_kmeans.predict(X_var)
centroids = var_kmeans.cluster_centers_
print("Sum of squared distances of samples to their closest cluster center for attrs with smallest variance:", var_kmeans.inertia_)
fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'black'}
colors = map(lambda x: colmap[x+1], labels)
plt.scatter(spam_df_with_3_attrs[46], spam_df_with_3_attrs[50], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Cluster for smallest variance')
plt.show()

#######################################################################

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_chi)
    distortions.append(sum(np.min(cdist(X_chi, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for chi2 selection')
plt.show()

chi_kmeans = KMeans(n_clusters=2)
chi_kmeans.fit(X_chi)
labels = chi_kmeans.predict(X_chi)
# print(labels)
# print(y)
centroids = chi_kmeans.cluster_centers_
print("Sum of squared distances of samples to their closest cluster center for chi2 selection:", chi_kmeans.inertia_)
plt.figure(figsize=(10, 10))
plt.subplot(211)
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'black'}
colors = map(lambda x: colmap[x+1], y)
plt.scatter(X_kbest_features[0], X_kbest_features[1], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 3000)
plt.ylim(0, 4000)
plt.title('Original clusters after chi2 test')
# plt.show()

plt.subplot(212)
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'black'}
colors = map(lambda x: colmap[x+1], labels)
plt.scatter(X_kbest_features[0], X_kbest_features[1], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 3000)
plt.ylim(0, 4000)
plt.title('Clusters after chi2 test')
plt.show()
