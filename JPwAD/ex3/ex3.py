import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA


def load_breast_cancer_dataset():

    df = load_breast_cancer()

    return df


def print_some_info(df):
    print("There are", len(df.feature_names), "attributes in the dataset")
    print("Features:", df.feature_names)
    print("Labels:", df.target_names)
    print("Shape:", df.data.shape)


def split_data_into_training_and_test_part(df):

    labels = df['target']
    features = df['data']

    train, test, train_labels, test_labels = train_test_split(features,
                                                              labels,
                                                              test_size=0.20,
                                                              random_state=42)
    return train, test, train_labels, test_labels


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def classify(classifier, X_train, X_test, y_train, y_test):

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    accuracy = np.sum(y_test == predictions, axis=0) / X_test.shape[0]
    print("Accuracy on test data: %.2f%%" % (accuracy * 100))


def plot_learning_rate(mlp):
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate = 0.0001")
    plt.ylim([0.0, 1.0])
    plt.plot(mlp.loss_curve_)
    plt.show()


def plot_accuracy(X_train, X_test, y_train, y_test):
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 50
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    mlp = MLPClassifier(hidden_layer_sizes=(29, 29, 29), max_iter=300, warm_start=True)

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))

        epoch += 1

    """ Plot """
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()


def do_PCA(X_train, X_test):
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def get_attr_with_smallest_cov(df):
    features = pd.DataFrame(df['data'])



def main():
    breast_cancer_df = load_breast_cancer_dataset()
    print_some_info(breast_cancer_df)

    X_train, X_test, y_train, y_test = split_data_into_training_and_test_part(breast_cancer_df)
    X_train, X_test = scale_data(X_train, X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(29, 29, 29), max_iter=300, warm_start=True)
    classify(mlp, X_train, X_test, y_train, y_test)
    plot_learning_rate(mlp)
    plot_accuracy(X_train, X_test, y_train, y_test)


#### PCA ####
    X_train, X_test = do_PCA(X_train, X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(29, 29, 29), max_iter=300, warm_start=True)
    classify(mlp, X_train, X_test, y_train, y_test)
    plot_learning_rate(mlp)
    plot_accuracy(X_train, X_test, y_train, y_test)


#### 2 attributes with smalles covariance $$$$
    get_attr_with_smallest_cov(breast_cancer_df)




if __name__ == "__main__":
    main()
