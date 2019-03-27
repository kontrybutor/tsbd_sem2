
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, normaltest


def load_quakes():
    filename = 'quakes.csv'
    quakes_df = pd.read_csv(filename, sep=",", )
    quakes_df = quakes_df.drop('index', 1)
    return quakes_df


def load_births():
    filename = 'births.csv'
    births_df = pd.read_csv(filename, sep=",")
    births_df = births_df.drop('index', 1)
    return births_df


def plot_histogram(data):
    plt.figure(figsize=(12, 10))
    sns.set_style('darkgrid')
    ax = sns.distplot(data, kde=True,)
    ax.set(ylabel='count')
    plt.show()


def check_normality(data):
    stat, p = normaltest(data)  # test for normality
    print('statistics =', stat, 'p =', p)
    alpha = 0.05
    if p < alpha:
        print('Samples are normally distributed, the null hypothesis can be rejected')
    else:
        print('Samples are not normally distributed, the null hypothesis cannot be rejected')


def test_hypothesis(data, hypothesis):
    stat, p = ttest_1samp(a=data, popmean=hypothesis)
    alpha = 0.05
    if p < alpha:
        print('Reject null hypothesis, mean is less than', hypothesis)
    else:
        print('Accept null hypothesis')


def main():
    depths = load_quakes()['depth']

    print("Normality test for depths...")
    check_normality(depths)
    plot_histogram(depths)
    print("Mean is: ", depths.mean())
    print("Test hypothesis for depths...")
    test_hypothesis(depths, 10000.0)

    print()

    births = load_births()['births']
    print("Normality test for births...")
    check_normality(births)
    plot_histogram(births)
    print("Mean is: ", births.mean())
    print("Test hypothesis for births...")
    test_hypothesis(births, 300.0)


if __name__ == "__main__":
    main()
