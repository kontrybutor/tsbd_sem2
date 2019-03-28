
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, normaltest, mannwhitneyu, shapiro
from statsmodels.stats.weightstats import ztest

births_filename = 'births.csv'
quakes_filename = 'quakes.csv'


def load_dataset(filename):
    df = pd.read_csv(filename, sep=",")
    df = df.drop('index', 1)
    return df


def plot_histogram(data):
    plt.figure(figsize=(12, 10))
    sns.set_style('darkgrid')
    ax = sns.distplot(data, kde=True,)
    ax.set(ylabel='count')
    plt.show()


def check_normality(data):
    stat, p = normaltest(data)  # test for normality
    alpha = 0.05
    if p < alpha:  # null hypothesis: data comes from a normal distribution
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')


def test_hypothesis(data, hypothesis):
    # stat, p = ttest_1samp(a=data, popmean=hypothesis)
    # stat, p = mannwhitneyu(data, hypothesis)
    stat, p = ztest(data, value=hypothesis)
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
    print("Mean is: ", births.mean())
    print("NULL HYPOTHESIS: Average of daily births is 10000")
    test_hypothesis(births, [10000])


if __name__ == "__main__":
    main()
