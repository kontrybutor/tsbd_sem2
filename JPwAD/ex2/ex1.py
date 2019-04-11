import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from fancyimpute import KNN


#  https://github.com/mohitpawar473/USA-Housing-Dataset


def load_dataset(filename):
    df = pd.read_csv(filename, sep=",")
    df = df.drop('Address', 1)

    return df


def compute_correlation(data, plot=False):
    corr = data.corr().abs()
    if plot:
        plot_corr_matrix(corr)


def plot_corr_matrix(data):
    plt.figure(figsize=(8, 7))

    sns.heatmap(data,
                annot=True,
                cmap=plt.cm.Reds,
                xticklabels=data.columns,
                yticklabels=data.columns)
    plt.show()


def print_stats(df, show=False):
    if show:
        print(df.head(10))
        print(df.describe())
        print()
        print(df.isnull().sum())

        values_count = df.count().sum()
        print("There are", values_count, "values in total")
        missed_values = df.isnull().sum().sum()
        print("There are", missed_values, "missed values in dataset")
        missed_values_ratio = missed_values / values_count
        print("Ratio between missed and total values is", missed_values_ratio)


def generate_missing_values(df, ratio):
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    for row, col in random.sample(ix, int(round(ratio * (len(ix))))):
        df.iat[row, col] = np.nan

    return df


def visualize_data(df):
    plt.figure(1, figsize=(15, 15))
    sns.pairplot(df)
    plt.show()

    plt.figure(2, figsize=(12, 12))
    sns.distplot(df['Price'])
    plt.show()


def drop_nan_values(df):
    return df.dropna()


def do_linear_regression(df, plot=True):
    x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population']]
    # x = df[['Avg. Area Income']]  # most correlated attribute
    y = df[['Price']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    predictions = lm.predict(x_test)

    if plot:
        plt.figure(figsize=(12, 12))
        plt.scatter(y_test, predictions, marker='.', c='black')

        plt.show()

    print('Coefficients:', lm.coef_)
    print('Intercept:', lm.intercept_)
    print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, predictions)))
    print('Variance score: {:.2f}'.format(r2_score(y_test, predictions)))


def fill_missing_values_with_interpolated_values(df):
    return df.interpolate(method='linear', limit_direction='forward')


def fill_missing_values_with_mean(df):
    imputer = SimpleImputer()
    data_with_imputed_values = pd.DataFrame(imputer.fit_transform(df))
    data_with_imputed_values.columns = df.columns

    return data_with_imputed_values


def fill_missing_values_with_linear_regression_coefs(df):
    df['Price'] = df.apply(lambda x:
                           (2.15992315e+01 * x['Avg. Area Income']
                            + 1.63747956e+05 * x['Avg. Area House Age']
                            + 1.22657149e+05 * x['Avg. Area Number of Rooms']
                            - 3.19517038e+03 * x['Avg. Area Number of Bedrooms']
                            + 1.54755246e+01 * x['Area Population']
                            - 2642580.58)
                           if pd.isnull(x['Price']) else x['Price'], axis=1)

    return df


def fill_missing_values_with_hot_deck(df):
    knn_filled_df = KNN(k=3).fit_transform(df)
    knn_filled_df = pd.DataFrame(knn_filled_df)
    knn_filled_df.columns = df.columns
    return knn_filled_df


def main():
    filename = 'USA_Housing.csv'
    usa_housing_df = load_dataset(filename)

    print_stats(usa_housing_df)
    usa_housing_df_with_nans = generate_missing_values(usa_housing_df, ratio=0.07)
    print_stats(usa_housing_df_with_nans, show=True)
    # compute_correlation(usa_housing_df_with_nans, True)

    # WITH KNN FILLING #
    print("WITH KNN FILLING")
    usa_housing_df_knn_filled = fill_missing_values_with_hot_deck(usa_housing_df_with_nans)
    print_stats(usa_housing_df_knn_filled, show=True)
    do_linear_regression(usa_housing_df_knn_filled, plot=False)

    # ROWS WITH NANS DROPPED #
    print("ROWS WITH NANS DROPPED")
    usa_housing_df_without_nans = drop_nan_values(usa_housing_df_with_nans)
    print_stats(usa_housing_df_without_nans, True)
    do_linear_regression(usa_housing_df_without_nans, plot=False)
    # visualize_data(usa_housing_df_without_nans)

    # REPLACED NANS WITH MEAN #
    print("REPLACED NANS WITH MEAN")
    usa_housing_df_mean_imputed = fill_missing_values_with_mean(usa_housing_df_with_nans)
    print_stats(usa_housing_df_mean_imputed, show=True)
    do_linear_regression(usa_housing_df_mean_imputed, plot=False)

    # INTERPOLATED #
    print("INTERPOLATED")
    usa_housing_df_interpolated = fill_missing_values_with_interpolated_values(usa_housing_df_with_nans)
    print_stats(usa_housing_df_interpolated, show=True)
    do_linear_regression(usa_housing_df_interpolated, plot=False)

    # WITH COEFS FROM LINEAR REGRESSION #
    print("WITH COEFS FROM LINEAR REGRESSION")
    usa_housing_df_filled_with_regression = fill_missing_values_with_linear_regression_coefs(usa_housing_df_with_nans)
    usa_housing_df_filled_with_regression.dropna(inplace=True)  # Z jakiegoś powodu zostaje niewielka liczba NaN
    print_stats(usa_housing_df_filled_with_regression, show=False)
    do_linear_regression(usa_housing_df_filled_with_regression)


if __name__ == "__main__":
    main()
