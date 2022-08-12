import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import ML_analysis.Feature_Engineering.utils.graphics as graph
from ML_analysis.Feature_Engineering.ML.FCBF import FCBF
from ML_analysis.Feature_Engineering.ML.__mRMR import mrmr_algorithm


class FeatureSelection:
    # TODO: add feature selection based on correlation matrix, PCA
    def __init__(self, data, label, feature_names, verbose: bool = False):
        """

        :param data: matrix of shape n_samples x n_features
        :param label: vector of shape n_samples x n_features
        :param feature_names: list of names of the features (for interpretability). Shape needs to be the same as label
        :param verbose: Whether to print the features selected or not
        """
        self.feature_names = feature_names
        self.data = data
        self.label = label
        self.verbose = verbose

    def random_forest_selection(self, nb_features, save_path: str = None):
        """
        Apply variance threshold, and then rank the features according to feature importance of Random Forest

        :param save_path: If not None, will save a feature importance figure to the path
        :param nb_features: Number of features to keep
        :return: Names and scores of the most important features.
        """
        selector = VarianceThreshold()
        new_data = selector.fit_transform(self.data)
        new_features = selector.transform([self.feature_names])[0]

        rf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=2, max_features=None, bootstrap=True,
                                    criterion='entropy')

        rf.fit(new_data, self.label)
        importances = rf.feature_importances_

        return self.__get_ranking(importances, nb_features, new_features, save_path)

    def shap_selection(self, nb_features, save_path: str = None):
        """
        Applying SHAP feature selection, according to https://shap.readthedocs.io/en/latest/

        :param nb_features: Number of features to keep
        :param save_path: If not None, will save a feature importance figure to the path
        :return: Names and scores of the most important features.
        """
        rf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=2, max_features=None, bootstrap=True,
                                    criterion='entropy')

        rf.fit(self.data, self.label)

        shap_values = shap.TreeExplainer(rf).shap_values(X=self.data, y=self.label)
        shap_values = np.array(shap_values)
        shap_values = np.abs(np.mean(shap_values[0], axis=0)) + np.abs(np.mean(shap_values[1], axis=0))

        return self.__get_ranking(shap_values, nb_features, self.feature_names, save_path)

    def chi2_selection(self, nb_features):
        """
        Use SelectKBest of sklearn

        :param nb_features: Number of features to keep
        :return: Name of the most important features
        """
        data_new = MinMaxScaler().fit_transform(X=self.data, y=self.label)

        selector = SelectKBest(chi2, k=nb_features)
        selector.fit(data_new, y=self.label)
        return selector.transform([self.feature_names])[0]

    def mutual_info_selection(self, nb_features, save_path: str = None):
        """

        :param nb_features: Number of features to keep
        :param save_path: If not None, will save a feature importance figure to the path
        :return: Names and scores of the most important features.
        """
        scores = mutual_info_classif(self.data, self.label)
        return self.__get_ranking(scores, nb_features, self.feature_names, save_path)

    def __RFE_selection(self, nb_features):
        """
        # TODO: fix, takes too much time to run

        :param nb_features: Number of features to keep
        :return: Name of the most important features.
        """
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=nb_features, step=10)
        selector = selector.fit(self.data, self.label)

        return self.feature_names[selector.support_]

    def lasso_selection(self, nb_features, save_path: str = None):
        """
        Feature selection based on LassoCV classifier

        :param nb_features: Number of features to keep
        :param save_path: If not None, will save a feature importance figure to the path
        :return: Names and scores of the most important features.
        """
        clf = LassoCV().fit(self.data, self.label)
        importances = np.abs(clf.coef_)

        return self.__get_ranking(importances, nb_features, self.feature_names, save_path)

    def mRMR(self, nb_features, method: str = "MID", save_path: str = None):
        """
        Apply mRMR algorithm, introduced by https://ieeexplore.ieee.org/document/1453511. Different methods defined in https://arxiv.org/pdf/1908.05376.pdf

        :param nb_features: Number of features to keep
        :param method: Method to use. Can be MID, MIQ, FCD, FCQ, RFCD, RFCQ or CFS
        :param save_path: If not None, will save a feature importance figure to the path
        :return: Names and scores of the most important features.
        """
        selector = VarianceThreshold()
        new_data = selector.fit_transform(self.data)
        new_features = selector.transform([self.feature_names])[0]

        selected_indices, all_scores = mrmr_algorithm(new_data, self.label, nb_features, method)
        new_features = new_features[selected_indices]

        return self.__get_ranking(all_scores, nb_features, new_features, save_path)

    def FCBF(self, delta=0.05, save_path: str = None):
        # TODO: Check this + introduce correlation instead of SU.
        """
        Apply FCBF algorithm, introduced by https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf.

        :param delta: threshold for SU score.
        :param save_path: If not None, will save a feature importance figure to the path
        :return: Names and scores of the most important features.
        """
        S_list = FCBF(self.data, self.label, self.feature_names, delta=delta)

        features_name_sorted = [x[2] for x in S_list]
        features_name_sorted = np.array(features_name_sorted)

        scores = [x[1] for x in S_list]
        scores = np.array(scores)

        return self.__get_ranking(scores, len(scores), features_name_sorted, save_path)

    def __get_ranking(self, scores, nb_features, features_name, save_path: str = None):
        indices = np.argsort(scores)

        scores_sorted = scores[indices]
        scores_sorted = scores_sorted[-nb_features:]

        features_name_sorted = features_name[indices]
        features_name_sorted = features_name_sorted[-nb_features:]

        if self.verbose is True:
            print("Features selected are", features_name_sorted)
        if save_path is not None:
            FeatureSelection.__visualization(features_name_sorted, scores_sorted, save_path)
        return features_name_sorted, scores_sorted, indices

    @staticmethod
    def __visualization(features_sorted, scores_sorted, save_path):
        fig, axes = graph.create_figure(subplots=(1, 1), tight_layout=True, figsize=(8, 12))
        axes[0][0].barh(features_sorted, scores_sorted)
        axes[0][0].set_xscale('log')

        graph.complete_figure(fig, axes, put_legend=[[False]],
                              savefig=True, legend_fontsize=20, frameon=[True],
                              main_title=save_path,
                              x_titles=[['importance']],
                              xticks_fontsize=15, yticks_fontsize=15,
                              xlabel_fontsize=18, ylabel_fontsize=18)
