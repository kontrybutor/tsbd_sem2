
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, normaltest, mannwhitneyu, shapiro, norm

births_filename = 'births.csv'
quakes_filename = 'quakes.csv'


def load_dataset(filename):
    df = pd.read_csv(filename, sep=",")
    df = df.drop('index', 1)
    return df


def plot_histogram(data):
    plt.figure(figsize=(12, 10))
    sns.set_style('darkgrid')
    ax = sns.distplot(data, kde=False,)
    ax.set(ylabel='count')
    plt.show()


def plot_histogram2(data):
    births_mean = data.mean()
    births_std = np.std(data)
    plt.figure()
    plt.hist(data, density=True)
    x = np.linspace(data.min(), data.max(), 1000)
    y = norm.pdf(x, births_mean, births_std)
    plt.plot(x, y)
    plt.plot(10000, norm.pdf(10000, births_mean, births_std), marker='o', markersize=3, color="red")
    plt.legend(('Gauss approximation', 'Point', 'Births'))
    plt.title('Histogram')
    plt.xlabel('Births')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.show()


def check_normality(data):
    stat, p = normaltest(data)  # test for normality
    alpha = 0.05
    if p < alpha:  # null hypothesis: data comes from a normal distribution
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')


def test_hypothesis(data, hypothesis):
    stat, p = ttest_1samp(a=data, popmean=hypothesis)
    alpha = 0.05
    if p < alpha:
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')


def main():
    depths = load_dataset(quakes_filename)['depth']

    print("NULL HYPOTHESIS: Depths are normally distributed")
    check_normality(depths)
    plot_histogram(depths)
    print("Mean is: ", depths.mean())
    print("NULL HYPOTHESIS: Average of quake's depth is 300m")
    test_hypothesis(depths, 300)

    print('-------------------------')

    births = load_dataset(births_filename)['births']
    print("NULL HYPOTHESIS: Births are normally distributed")
    check_normality(births)
    plot_histogram(births)
    plot_histogram2(births)
    print("Mean is: ", births.mean())
    print("NULL HYPOTHESIS: Average of daily births is 10000")
    test_hypothesis(births, 10000)


if __name__ == "__main__":
    main()
