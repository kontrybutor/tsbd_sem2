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

coef = 0
intercept = 0


def load_dataset(filename):
    df = pd.read_csv(filename, sep=",")

    df = df.drop(['Address', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                  'Avg. Area Number of Bedrooms', 'Area Population'], 1)

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

        # print(df.loc[[1536]])

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

    # df.iat[1536, 0] = np.nan

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


def do_linear_regression(df, plot=True, plot_title=""):
    x = df[['Avg. Area Income']]  # most correlated attribute
    y = df[['Price']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    predictions = lm.predict(x_test)

    global coef, intercept
    coef = lm.coef_[0]
    intercept = lm.intercept_

    if plot:
        plt.figure(figsize=(12, 12))
        plt.scatter(x_test, y_test, marker='.', c='black')
        plt.plot(x_test, predictions, color='blue', linewidth=3)
        plt.ylabel('Price')
        plt.xlabel('Avg. Area Income')
        plt.title(plot_title)
        plt.show()

    print('Coefficients:', lm.coef_)
    print('Intercept:', lm.intercept_)
    print('Mean squared error: {:.2f}'.format(mean_squared_error(y_test, predictions)))
    print('Variance score: {:.2f}'.format(r2_score(y_test, predictions)))


def fill_missing_values_with_interpolated_values(df):
    return df.interpolate(method='linear', limit_direction='both')


def fill_missing_values_with_mean(df):
    imputer = SimpleImputer()
    data_with_imputed_values = pd.DataFrame(imputer.fit_transform(df))
    data_with_imputed_values.columns = df.columns

    return data_with_imputed_values


def fill_missing_values_with_linear_regression_coefs(df):
    for index, row in df.iterrows():
        if pd.isnull(row['Price'].item()):
            df.at[index, 'Price'] = (coef * row['Avg. Area Income'] + intercept)
        elif pd.isnull(row['Avg. Area Income'].item()):
            df.at[index, 'Avg. Area Income'] = ((row['Price'] - intercept) / coef)
    return df


def fill_missing_values_with_hot_deck(df):
    knn_filled_df = KNN(k=3).fit_transform(df)
    knn_filled_df = pd.DataFrame(knn_filled_df)
    knn_filled_df.columns = df.columns
    return knn_filled_df


def main():
    filename = 'USA_Housing.csv'
    usa_housing_df = load_dataset(filename)
    print("RAW DATASET")
    print_stats(usa_housing_df, show=True)

    print("INITIAL DATASET WITH MISSING VALUES")
    usa_housing_df_with_nans = generate_missing_values(usa_housing_df, ratio=0.23)
    print_stats(usa_housing_df_with_nans, show=True)
    # compute_correlation(usa_housing_df_with_nans, True)
    # visualize_data(usa_housing_df_with_nans)

    print("ROWS WITH NANS DROPPED")
    usa_housing_df_without_nans = drop_nan_values(usa_housing_df_with_nans)
    print_stats(usa_housing_df_without_nans, show=True)
    do_linear_regression(usa_housing_df_without_nans, plot=True,
                         plot_title="Rows with missing values dropped")

    print("WITH KNN FILLING")
    usa_housing_df_knn_filled = fill_missing_values_with_hot_deck(usa_housing_df_with_nans)
    print_stats(usa_housing_df_knn_filled, show=True)
    do_linear_regression(usa_housing_df_knn_filled, plot=True,
                         plot_title="Missing values replaced by hot-deck method")

    print("REPLACED NANS WITH MEAN")
    usa_housing_df_mean_imputed = fill_missing_values_with_mean(usa_housing_df_with_nans)
    print_stats(usa_housing_df_mean_imputed, show=True)
    do_linear_regression(usa_housing_df_mean_imputed, plot=True,
                         plot_title="Missing values replaced by mean")

    print("INTERPOLATED")
    usa_housing_df_interpolated = fill_missing_values_with_interpolated_values(usa_housing_df_with_nans)
    print_stats(usa_housing_df_interpolated, show=True)
    do_linear_regression(usa_housing_df_interpolated, plot=True,
                         plot_title="Interpolated values")

    print("WITH COEFS FROM LINEAR REGRESSION")
    usa_housing_df_filled_with_regression = fill_missing_values_with_linear_regression_coefs(usa_housing_df_with_nans)
    usa_housing_df_filled_with_regression.dropna(inplace=True)  # Z jakiegoś powodu zostaje niewielka liczba NaN
    print_stats(usa_housing_df_filled_with_regression, show=True)
    do_linear_regression(usa_housing_df_filled_with_regression, plot=True,
                         plot_title="With values from linear regression")


if __name__ == "__main__":
    main()
