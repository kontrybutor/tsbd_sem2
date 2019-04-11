
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_1samp, normaltest, mannwhitneyu, norm

births_filename = 'births.csv'


def load_dataset(filename):
    df = pd.read_csv(filename, sep=",")
    df = df.drop('index', 1)

    return df


def plot_histogram(data):
    plt.figure(figsize=(12, 10))
    sns.set_style('darkgrid')
    mu, std = norm.fit(data)
    ax = sns.distplot(data, kde=False, fit=norm)
    plt.plot(10000, norm.pdf(10000, mu, std), marker='o', markersize=3, color="red")
    ax.set(ylabel='count')
    plt.legend(('Gauss approximation', 'Point', 'Births'))
    plt.title('Histogram')
    plt.xlabel('Births')
    plt.ylabel('Amount')
    plt.show()


def check_normality(data):
    stat, p = normaltest(data)  # test for normality
    alpha = 0.05
    print("p-value = {}".format(p))
    if p < alpha:  # null hypothesis: data comes from a normal distribution
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')


def test_hypothesis(data, hypothesis):
    print("Test t-Studenta:")
    alpha = 0.05
    stat, p = ttest_1samp(a=data, popmean=hypothesis)
    print("p-value = {}".format(p))
    if p < alpha:
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')

    print("Test U-Manna-Whitneya:")
    # random = np.random.normal(10000, 0.1, 10000)
    # stat, p = mannwhitneyu(data, random)
    stat, p = mannwhitneyu(data, [hypothesis])
    print("p-value = {}".format(p))
    if p < alpha:
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')


def main():
    births = load_dataset(births_filename).get('births')
    print("Null hypothesis: data comes from a normal distribution")
    check_normality(births)
    print("--------------------------------")
    plot_histogram(births)
    print("Null hypothesis: Average of daily births is 10000")
    test_hypothesis(births, 10000)


if __name__ == "__main__":
    main()
