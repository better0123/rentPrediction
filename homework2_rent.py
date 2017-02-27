"""This is the main module containing the data processing and model training code used for hw2
"""
__author__ = "Emily"

import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
if sys.version_info[0] < 3:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import io
from data import to_be_removed, to_be_binary, binary_dic, categorical_dic, numerical_dic


def download_data():
    r"""Download the data from internet and parse to pandas' data frame

    Returns
    -------
    data : DataFrame
        Data downloaded and parsed

    """
    target_url = "https://ndownloader.figshare.com/files/7586326"
    url_content = requests.get(target_url).content
    return pd.read_csv(io.StringIO(url_content.decode('utf-8')))


def drop_missing_value(dataframe):
    r"""Drop variables that have missing value only

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe needed to be processed

    Returns
    -------
    dataframe : DataFrame
        Processed data frame
    all_missing : array_like
        Missing data

    """
    all_missing = []
    for col in dataframe.columns:
        if len(dataframe[col].unique()) == 1 and np.isnan(dataframe[col].unique()[0]):
            all_missing.append(col)

    # drop columns with all missing data
    dataframe = dataframe.drop(all_missing, inplace=False, axis=1)
    return dataframe, all_missing


def impute(X_train, X_test, strategy):
    r"""Impute training data and transform test data based on training data specs

    Parameters
    ----------
    X_train : DataFrame
        Training dataset
    X_test : DataFrame
        Testing dataset
    strategy : Strategy
        Strategy used to impute data

    Returns
    -------
    X_train_imputed : DataFrame
        Training dataset with missing value imputed
    X_test_imputed : DataFrame
        Testing dataset with missing value imputed

    """
    imp = Imputer(missing_values=np.nan, strategy=strategy).fit(X_train)
    X_train_imputed = imp.transform(X_train)
    X_train_imputed = pd.DataFrame(
        X_train_imputed, columns=X_train.columns)
    X_test_imputed = imp.transform(X_test)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    return X_train_imputed, X_test_imputed


def process_data(raw_data):
    r"""Process given dataset, remove invalid feature

    Parameters
    ----------
    raw_data : DataFrame
        The raw dataset to be processed

    Returns
    -------
    X_train : DataFrame
        Training data.
    X_test : array_like
        Testing data.
    y_train : DataFrame
        Training target.
    y_test : array_like
        Testing target.

    """
    # features with rent associalted information, needed to be removed
    df = raw_data.drop(to_be_removed, inplace=False, axis=1)

    # remove NA in uf17 (dependent variable)
    df = df[df['uf17'] != 99999]

    # turn features in to_be_binary into binary features
    for key in to_be_binary.keys():
        possible_values = to_be_binary[key].keys()
        df[key][~(df[key].isin(possible_values))] = 0

    #########replace no reply with np.nan for the binary variables##########
    binary_keys = binary_dic.keys()
    df_binary = df[binary_keys]
    for key in binary_keys:

        # possible_values : list of known-value
        possible_values = binary_dic[key].keys()

        # replace anything outside known-value with NaN
        df_binary[key][~(df_binary[key].isin(possible_values))] = np.nan

    #########replace no reply with np.nan for the categorical variables#######
    categorical_keys = categorical_dic.keys()
    df_categorical = df[categorical_keys]
    for key in categorical_keys:

        # possible_values : list of known-value
        possible_values = categorical_dic[key].keys()

        # replace anything outside known-value with NaN
        df_categorical[key][
            ~(df_categorical[key].isin(possible_values))] = np.nan

    #########replace no reply with np.nan for the numerical variables#########
    numerical_keys = numerical_dic.keys()
    df_numerical = df[numerical_keys]
    for key in numerical_keys:

        # maximum meaningful value
        possible_values = numerical_dic[key]

        # replace anything outside meaningful value with NaN
        df_numerical[key][df_numerical[key] > possible_values] = np.nan

    df_binary, binary_all_missing = drop_missing_value(df_binary)
    df_categorical, categorical_all_missing = drop_missing_value(
        df_categorical)
    df_numerical, numerical_all_missing = drop_missing_value(df_numerical)

    # split numerical data into train and test
    cols = [col for col in df_numerical.columns if col not in ['uf17']]
    data_numerical = df_numerical[cols]
    target = df['uf17']
    X, y = data_numerical, target
    X_train_nu, X_test_nu, y_train_nu, y_test_nu = train_test_split(
        X, y, random_state=0)

    # split binary data into train and test
    data_binary = df_binary
    X, y = data_binary, target
    X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(
        X, y, random_state=0)

    # split categorical data into train and test
    data_categorical = df_categorical
    X, y = data_categorical, target
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y, random_state=0)

    X_train_bi_imputed, X_test_bi_imputed = impute(
        X_train_bi, X_test_bi, 'most_frequent')
    X_train_cat_imputed, X_test_cat_imputed = impute(
        X_train_cat, X_test_cat, 'most_frequent')
    X_train_nu_imputed, X_test_nu_imputed = impute(
        X_train_nu, X_test_nu, 'median')

    # concatenate binary, categorical, numerical into the final dataframe
    X_train = pd.concat(
        [X_train_bi_imputed, X_train_cat_imputed, X_train_nu_imputed], axis=1)
    X_test = pd.concat(
        [X_test_bi_imputed, X_test_cat_imputed, X_test_nu_imputed], axis=1)

    # y_test_nu == y_test_bi == y_test_cat, same as y_train_*
    y_train, y_test = y_train_nu, y_test_nu

    # oneHot for the categorical data
    categorical_all_missing = ['uf10', 'uf9', 'sc120', 'sc144', 'sc141']
    for c in categorical_dic.keys():
        if c in categorical_all_missing:
            continue
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # dealing with train and test potential categorical feature inconsistency:
    # say variable feature_6 shows up only in the training dataset but not
    # test dataset, then it should be removed
    train_col = X_train.columns
    test_col = X_test.columns
    missing_categorical_train_cols = []
    missing_categorical_test_cols = []
    if len(test_col) < len(train_col):
        for train in train_col:
            if train not in test_col:
                missing_categorical_test_cols.append(train)
    for test in test_col:
        if test not in train_col:
            missing_categorical_train_cols.append(test)

    # drop columns not exist in test data
    X_train = X_train.drop(missing_categorical_test_cols,
                           inplace=False, axis=1)

    # drop columns not exist in train data
    X_test = X_test.drop(missing_categorical_train_cols, inplace=False, axis=1)
    return X_train, X_test, y_train, y_test


def feature_selection(X_train, X_test, y_train, y_test):
    r"""Select features from given datasets

    Parameters
    ----------
    X_train : DataFrame
        Training dataset
    X_test : array_like
        Testing dataset
    y_train : DataFrame
        Training target
    y_test : array_like
        Testing target

    Returns
    -------
    X_train : DataFrame
        Training dataset
    X_test : array_like
        Testing dataset
    y_train : DataFrame
        Training target
    y_test : array_like
        Testing target

    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso = LassoCV().fit(X_train_scaled, y_train)
    zero_coef_bol = lasso.coef_ == 0
    zero_coef = []
    for i in range(len(zero_coef_bol)):
        if zero_coef_bol[i] == True:
            zero_coef.append(X_train.columns[i])

    # remove features with zero lasso coef
    X_train = X_train.drop(zero_coef, inplace=False, axis=1)
    X_test = X_test.drop(zero_coef, inplace=False, axis=1)
    return X_train, X_test, y_train, y_test


def predict_rent(X_train, X_test, y_train, y_test):
    r"""Predict rent data using given datasets

    Parameters
    ----------
    X_train : DataFrame
        Training dataset
    X_test : array_like
        Testing dataset
    y_train : DataFrame
        Training target
    y_test : array_like
        Testing target

    Returns
    -------
    X_train : DataFrame
        Training dataset
    X_test : array_like
        Testing dataset
    predicted : array_like
        Predicted data using LASSO

    """
    clf = Ridge(alpha=110)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return X_test, y_test, predicted


def score_rent():
    r"""Score the rent model trained with R2

    Returns
    -------
    Rs : number
        R Square score

    """
    X_train, X_test, y_train, y_test = process_data(download_data())

    X_train, X_test, y_train, y_test = feature_selection(
        X_train, X_test, y_train, y_test)

    X_test, y_test, predicted = predict_rent(X_train, X_test, y_train, y_test)
    Rs = r2_score(y_test, predicted)
    print('R Square: ', Rs)
    return Rs

if __name__ == "__main__":
    score_rent()
