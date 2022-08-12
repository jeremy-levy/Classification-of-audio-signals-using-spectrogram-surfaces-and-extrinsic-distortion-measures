import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import numpy as np

from ML_analysis.Feature_Engineering.utils.help_classes import ScalerEnum


class DataFromCSV:
    # TODO: finish get_data_stratify, from COPD classifier
    # TODO: add option of feature selection step before splitting
    def __init__(self, filename: str, label: str, features_to_remove: list = None, test_size: float = 0.2,
                 scaler_technique: ScalerEnum = None, median_imputation: bool = True):
        """
        Get data from a csv file.

        :param filename: csv file with the data
        :param label: Name of the column with the label within the data file
        :param features_to_remove: List of features to remove
        :param test_size: Percentage of the test size
        :param scaler_technique: Technique for preprocessing data to use. See ScalerEnum for the options. If None, MaxAbs is used.
        :param median_imputation: whether to use median imputation or not

        """
        if features_to_remove is None:
            features_to_remove = []

        data = pd.read_csv(filename)

        y = data[label]
        X = data.drop(features_to_remove + [label], axis=1)
        self.feature_names = X.columns

        if median_imputation is True:
            X = self.median_imputation(X)

        self.data_class = Data(X, y, test_size, scaler_technique)

    def get_data(self):
        """
        Perform the split of the data. Stratify with respect to the labels.

        :return: X_train, X_test, y_train, y_test, label encoder
        """
        return self.data_class.get_data()

    @staticmethod
    def median_imputation(X):
        """
        Performs median imputation, for each column of X independently.

        :param X: pandas dataframe of shape n_samples x n_features
        :return: pandas dataframe with same shape, after median imputation
        """
        for feature in X.columns:
            median_feature = np.nanmedian(X[feature])
            X[feature] = X[feature].fillna(median_feature)
        return X


class Data:
    def __init__(self, data, label, test_size: float = 0.2, scaler_technique: ScalerEnum = None):
        """

        :param data: matrix of shape n_samples x n_features
        :param label: matrix of shape n_samples x n_classes
        :param test_size: Percentage of the test size
        :param scaler_technique: Technique for preprocessing data to use. See ScalerEnum for the options. If None, MaxAbs is used.
        """
        self.label = label
        self.le, self.label = self.__encode_label()

        self.data = data
        self.scaler_technique = scaler_technique
        self.test_size = test_size

    def __encode_label(self):
        le = LabelEncoder()
        le.fit(self.label)
        return le, le.transform(self.label)

    def __preprocessing_data(self, X_train, X_test):
        scaler = None
        if self.scaler_technique is None:
            return X_train, X_test, scaler
        if self.scaler_technique == ScalerEnum.min_max:
            scaler = MinMaxScaler()
        elif self.scaler_technique == ScalerEnum.max_abs:
            scaler = MaxAbsScaler()
        elif self.scaler_technique == ScalerEnum.standard:
            scaler = StandardScaler()
        elif self.scaler_technique == ScalerEnum.robust:
            scaler = RobustScaler()

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler

    def get_data(self):
        """
        Perform the split of the data. Stratify with respect to the labels.

        :return: X_train, X_test, y_train, y_test, label encoder
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=self.test_size,
                                                            stratify=self.label)
        X_train, X_test, _ = self.__preprocessing_data(X_train, X_test)

        return X_train, X_test, y_train, y_test, self.le
