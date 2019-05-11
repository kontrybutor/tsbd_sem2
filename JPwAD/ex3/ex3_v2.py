import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

filename = 'absenteeism.csv'

names = ['ID', 'Reason', 'Month of absence',
         'Day of the week', 'Seasons',
         'Transportation expense', 'Distance to Work', 'Service time', 'Age', 'Work load Average/day',
         'Hit target', 'Disciplinary failure', 'Education', 'Son', 'Social drinker',
         'Social smoker', 'Pet', 'Weight', 'Height', 'BMI', 'Absenteeism time in hours']

absenteeism_df = pd.read_csv(filename, sep=";", names=names)

absenteeism_df = absenteeism_df[:].astype('float64')

absenteeism_df = absenteeism_df.drop('ID', 1)
# print(absenteeism_df.head(5))
# print(absenteeism_df.describe())
#
X = absenteeism_df.drop('Absenteeism time in hours', 1)
y = absenteeism_df['Absenteeism time in hours']
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
#
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)
#
#
# n_iters = [1, 10, 20, 50, 100, 250]
# scores = []
# for n_iter in n_iters:
#     model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
#     model.fit(X_train_std, y_train)
#     scores.append(model.score(X_test_std, y_test))
#
# plt.title("Effect of n_iter for SGD before PCA")
# plt.xlabel("n_iter")
# plt.ylabel("Accuracy score")
# plt.ylim([0.15, 0.5])
# plt.plot(n_iters, scores)
# plt.show()
#
# print("SGD Accuracy before PCA:", model.score(X_test_std, y_test))
#
# pca = PCA(n_components=2)
# pca.fit(X_train_std)
# X_train_reduced = pca.transform(X_train_std)
# X_test_reduced = pca.transform(X_test_std)
#
# n_iters = [50, 100, 150, 200, 250]
# scores = []
# for n_iter in n_iters:
#     model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
#     model.fit(X_train_reduced, y_train)
#     scores.append(model.score(X_test_reduced, y_test))
#
# plt.title("Effect of n_iter for SGD after PCA")
# plt.xlabel("n_iter")
# plt.ylabel("Accuracy score")
# plt.ylim([0.15, 0.5])
# plt.plot(n_iters, scores)
# plt.show()
#
# print("SGD Accuracy after PCA:", model.score(X_test_reduced, y_test))
#
#
# absenteeism_df_without_target = absenteeism_df.drop('Absenteeism time in hours', 1)
#
# # print(absenteeism_df.var().sort_values())
#
#
# reduced_absence_df = absenteeism_df[['Absenteeism time in hours', 'Disciplinary failure', 'Social smoker']]
# # reduced_absence_df = absenteeism_df[['Absenteeism time in hours', 'Weight', 'BMI']]
#
# reduced_data = reduced_absence_df.values
# print(reduced_absence_df.head())
# X_red = reduced_data[:, 1:]  # all rows, no label
# y_red = reduced_data[:, 0]
#
# X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.2, random_state=0)
#
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)
#
# n_iters = [50, 100, 150, 200, 250]
# scores = []
# for n_iter in n_iters:
#     model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter, tol=1e-3)
#     model.fit(X_train_std, y_train)
#     scores.append(model.score(X_test_std, y_test))
#
# plt.title("Effect of n_iter for SGD for 2 attr with the smallest variance")
# plt.xlabel("n_iter")
# plt.ylabel("Accuracy score")
# plt.ylim([0.15, 0.5])
# plt.plot(n_iters, scores)
# plt.show()
#
# print("SGD Accuracy for 2 attr with the smallest variance:", model.score(X_test_std, y_test))


chi2_features = SelectKBest(chi2, k=4)
X_kbest_features = chi2_features.fit_transform(X, y)
print(pd.DataFrame(X_kbest_features))
print('Original feature number:', X.shape[1])
print('Reduced feature number:', X_kbest_features.shape[1])
chi2_df = pd.DataFrame(X_kbest_features)
chi2_df.columns = ['Reason', 'Transportation expense', 'Work load Average/day', 'Disciplinary failure']
print(pd.DataFrame(X_kbest_features))
chi2_df['Absenteeism time in hours'] = absenteeism_df['Absenteeism time in hours']


chi_data = chi2_df.values
X_red = chi_data[:, 1:]  # all rows, no label
y_red = chi_data[:, 0]
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

plt.title("Effect of n_iter for SGD for attrs from chi2 test")
plt.xlabel("n_iter")
plt.ylabel("Accuracy score")
plt.ylim([0.15, 0.5])
plt.plot(n_iters, scores)
plt.show()


print("SGD Accuracy after chi2 test", model.score(X_test_std, y_test))




