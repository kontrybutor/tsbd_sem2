import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


filename = 'walkers.csv'
names = ['date', 'time', 'username', 'activity', 'acceleration_x',
         'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
walkers_df = pd.read_csv(filename, sep=',', names=names)
walkers_df = walkers_df.drop(["date", "time", "username"], axis=1)
print(walkers_df.head())
print(walkers_df.describe())

data = walkers_df.values
X = data[:, 1:]  # all rows, no label
y = data[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# knn = KNeighborsClassifier(n_neighbors=27)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
#
# print("KNN Accuracy:", accuracy_score(y_test, y_pred))

n_iters = [1, 10, 20, 50, 100, 250]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
    model.fit(X_train_std, y_train)
    scores.append(model.score(X_test_std, y_test))

plt.title("Effect of n_iter for SGD before PCA")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.ylim([0.8, 0.95])
plt.plot(n_iters, scores)
plt.show()

print("SGD Accuracy before PCA:", model.score(X_test_std, y_test))


pca = PCA(n_components=2)
pca.fit(X_train_std)
X_train_reduced = pca.transform(X_train_std)
X_test_reduced = pca.transform(X_test_std)

n_iters = [1, 10, 20, 50, 100, 250]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
    model.fit(X_train_reduced, y_train)
    scores.append(model.score(X_test_reduced, y_test))

plt.title("Effect of n_iter for SGD after PCA")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.ylim([0.45, 0.7])
plt.plot(n_iters, scores)
plt.show()

print("SGD Accuracy after PCA:", model.score(X_test_reduced, y_test))


walkers_df_without_target = walkers_df.drop('activity', 1)
# corr = walkers_df_without_target.corr().abs()
# plt.figure(figsize=(16, 14))
# sns.heatmap(corr,
#             annot=True,
#             cmap=plt.cm.Reds,
#             xticklabels=corr.columns,
#             yticklabels=corr.columns)
# plt.show()


print(walkers_df_without_target.var().sort_values())
reduced_walkers_df = walkers_df[['activity', 'acceleration_z', 'acceleration_y']]


reduced_data = reduced_walkers_df.values
print(reduced_walkers_df.head())
X_red = reduced_data[:, 1:]  # all rows, no label
y_red = reduced_data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

n_iters = [50, 100, 150, 200, 250]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
    model.fit(X_train_std, y_train)
    scores.append(model.score(X_test_std, y_test))

plt.title("Effect of n_iter for SGD for 2 attr with the smallest variance")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
# plt.ylim([0.5, 0.8])
plt.plot(n_iters, scores)
plt.show()

print("SGD Accuracy for 2 attr with the smallest variance:", model.score(X_test_std, y_test))


# normalized_data = Normalizer().fit(X)
# normalized_data = normalized_data.transform(X)
#
# data = pd.DataFrame(normalized_data).values
# X = data[:, 1:]  # all rows, no label
# y = data[:, 0]
#
# print(pd.DataFrame(normalized_data).head())
#
#
# chi2_features = SelectKBest(chi2, k=2)
# X_kbest_features = chi2_features.fit_transform(X, y)
#
# print('Original feature number:', X.shape[1])
# print('Reduced feature number:', X_kbest_features.shape[1])
# print(X_kbest_features)


