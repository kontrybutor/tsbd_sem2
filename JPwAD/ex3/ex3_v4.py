import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


filename = 'spambase.data'
spam_df = pd.read_csv(filename, sep=',', header=None)
print(spam_df.head())
print(spam_df.describe())

# spam_df = spam_df.reindex(np.random.permutation(spam_df.index))
spam_df = shuffle(spam_df)
data = spam_df.values.astype('float64')
X = data[:, :-1]  # all rows, no label
y = data[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
#
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter,)
    model.fit(X_train_std, y_train)
    scores.append(model.score(X_test_std, y_test))
    losses.append(model.loss_)

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

print("MLP Accuracy before PCA:", model.score(X_test_std, y_test))


pca = PCA(n_components=2)
pca.fit(X_train_std)
X_train_reduced = pca.transform(X_train_std)
X_test_reduced = pca.transform(X_test_std)

n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter, )
    model.fit(X_train_reduced, y_train)
    scores.append(model.score(X_test_reduced, y_test))
    losses.append(model.loss_)

plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate after PCA")
plt.ylim([0.0, 1.0])
plt.plot(n_iters, losses)
plt.show()
plt.title("Effect of n_iter for MLP after PCA")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.plot(n_iters, scores)
plt.show()

print("MLP Accuracy after PCA:", model.score(X_test_reduced, y_test))


print(spam_df.var().sort_values())

new_spam_df = spam_df[[46, 50, 57]]
print(new_spam_df.var())

data = new_spam_df.values.astype('float64')
X_red = data[:, :-1]  # all rows, no label
y_red = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=n_iter, )
    model.fit(X_train_std, y_train)
    scores.append(model.score(X_test_std, y_test))
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
plt.plot(n_iters, scores)
plt.show()

print("MLP Accuracy with 2 attrs with smallest variance:", model.score(X_test_std, y_test))

chi2_features = SelectKBest(chi2, k=2)
X_kbest_features = chi2_features.fit_transform(X, y)

print('Original feature number:', X.shape[1])
print('Reduced feature number:', X_kbest_features.shape[1])
X_kbest_features = pd.DataFrame(X_kbest_features)
X_kbest_features[2] = spam_df[[57]]
data = X_kbest_features.values
X_chi = data[:, :-1]  # all rows, no label
y_chi = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_chi, y_chi, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300]
scores = []
losses = []
for n_iter in n_iters:
    model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=n_iter, )
    model.fit(X_train_std, y_train)
    scores.append(model.score(X_test_std, y_test))
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
plt.ylim([0.5, 0.65])
plt.plot(n_iters, scores)
plt.show()

print("MLP Accuracy after chi2 test:", model.score(X_test_std, y_test))


################################# CLUSTERING ################################
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300, 400, 1000]
#
# scores = []
# for n_iter in n_iters:
#     correct = 0
#     kmeans = KMeans(n_clusters=2, max_iter=n_iter)
#     kmeans = kmeans.fit(X_scaled)
#     for i in range(len(X)):
#         predict_me = np.array(X[i].astype(float))
#         predict_me = predict_me.reshape(-1, len(predict_me))
#         prediction = kmeans.predict(predict_me)
#         if prediction[0] == y[i]:
#             correct += 1
#     scores.append(correct/len(X))
#     print(correct/len(X))
#
#
#
# plt.title("Effect of n_iter for K-means clustering")
# plt.xlabel("n_iter")
# plt.ylabel("Accuracy score")
# # plt.ylim([0.5, 0.65])
# plt.plot(n_iters, scores)
# plt.show()
#
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train_reduced)
#
# n_iters = [1, 10, 20, 30, 50, 70, 85, 100, 150, 300, 400, 1000]
#
# scores = []
# for n_iter in n_iters:
#     correct = 0
#     kmeans = KMeans(n_clusters=2)
#     kmeans = kmeans.fit(X_scaled)
#     for i in range(len(X)):
#         predict_me = np.array(X[i].astype(float))
#         predict_me = predict_me.reshape(-1, len(predict_me))
#         prediction = kmeans.predict(predict_me)
#         if prediction[0] == y[i]:
#             correct += 1
#
#     print(correct/len(X))
#
#
#
# plt.title("Effect of n_iter for K-means clustering")
# plt.xlabel("n_iter")
# plt.ylabel("Accuracy score")
# # plt.ylim([0.5, 0.65])
# plt.plot(n_iters, scores)
# plt.show()


