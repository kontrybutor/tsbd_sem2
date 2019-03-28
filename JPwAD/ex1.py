import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'fertility_diagnosis.data'

names = ['Season', 'Age', 'Childish diseases', 'Accident', 'Surgical intervention',
         'High fevers', 'Alcohol consumption', 'Smoking habit', 'Sitting hours', 'Diagnosis']
fertility_df = pd.read_csv(filename, sep=",", header=None, names=names)


def print_statistics(attribute):  # cechy ilościowe
    attr = fertility_df[attribute]
    print("Median is:", attr.mean())
    print("Max value is:", attr.max())
    print("Min value is", attr.min())


def print_dominant_for_diagnosis():  # cecha jakościowa
    attr = fertility_df['Diagnosis']
    values = attr.value_counts()
    print("There are following values in Diagnosis attribute:")
    print(values)
    dominant = attr.value_counts().idxmax()
    print("The most common value is: {}".format(dominant))


def compute_correlation(plot=True):
    corr = fertility_df.corr().abs()
    if plot:
        plot_corr_matrix(corr)


def plot_corr_matrix(data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data,
                annot=True,
                cmap=plt.cm.Reds,
                xticklabels=data.columns,
                yticklabels=data.columns)
    plt.show()


def plot_histogram(data):
    plt.figure(figsize=(12, 10))
    sns.set_style('darkgrid')
    ax = sns.distplot(data, kde=False,)
    ax.set(ylabel='Count')
    plt.show()


def main():
    for name in names[:-1]:
        print("Statistics for:", name)
        print_statistics(name)
        print()
    print_dominant_for_diagnosis()
    compute_correlation()
    plot_histogram(fertility_df["Age"])
    plot_histogram(fertility_df['Sitting hours'])


if __name__ == "__main__":
    main()
