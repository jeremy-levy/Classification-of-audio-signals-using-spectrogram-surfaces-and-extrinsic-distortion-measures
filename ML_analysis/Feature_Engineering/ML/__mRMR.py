# Maximum Relevance and Minimum Redundancy
# https://arxiv.org/pdf/1908.05376.pdf

import math
import random
from math import log, e

import numpy as np
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score


def apply_mrmr(X, y, nb_features, method):
    num_features = len(X[0])

    if method == "MID" or method == "MIQ":
        all_scores = [mutual_info_score(X[:, i], y) for i in range(num_features)]
    elif method == "FCD" or method == "FCQ":
        all_scores = [f_oneway(X[:, i], y)[0] for i in range(num_features)]
    elif method == "RFCD" or method == "RFCQ":
        RF_clf = RandomForestClassifier(n_estimators=500)
        RF_clf.fit(X, y)
        all_scores = RF_clf.feature_importances_
    else:
        # Default case is MID
        all_scores = [mutual_info_score(X[:, i], y) for i in range(num_features)]
    mi_score_matrix = np.zeros((num_features, num_features))

    start_feature_index = np.argmax(all_scores)
    selected_indices_list = [start_feature_index]

    for _ in range(nb_features - 1):
        feature_score = []
        for i in range(num_features):
            if i in selected_indices_list:
                feature_score.append(-float('inf'))
            else:
                score = all_scores[i]
                diff = 0
                for j in selected_indices_list:
                    # Keep a symmetric matrix
                    if j > i:
                        if mi_score_matrix[i][j] == 0:
                            mi_score_matrix[i][j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        diff += mi_score_matrix[i][j]
                    else:
                        if mi_score_matrix[j][i] == 0:
                            mi_score_matrix[j][i] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        diff += mi_score_matrix[j][i]

                if method == "MID" or method == "FCD" or method == "RFCD":
                    feature_score.append(score - diff / len(selected_indices_list))
                elif method == "MIQ" or method == "FCQ" or method == "RFCQ":
                    feature_score.append(score / (diff / len(selected_indices_list)))
                else:
                    # Default case is MID
                    feature_score.append(score - diff / len(selected_indices_list))

        max_score_index = np.argmax(np.array(feature_score))
        selected_indices_list.append(max_score_index)

    all_scores = np.array(all_scores)
    all_scores = all_scores[selected_indices_list]
    return selected_indices_list, all_scores


def SU(x_i, x_j):
    def entropy(labels, base=None):
        """ Computes entropy of label distribution. """
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0
        ent = 0.
        # Compute entropy
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)
        return ent

    return 2 * (mutual_info_score(x_i, x_j) / (entropy(x_i) + entropy(x_j)))


def CFS(X, y, k):
    num_features = len(X[0])
    # rcf
    merit_feature_class = [SU(X[:, i], y) for i in range(num_features)]
    # rff
    rff = []
    for i in range(num_features):
        scores = []
        for _ in range(0, i + 1):
            scores.append(0)
        for j in range(i + 1, num_features):
            scores.append(SU(X[:, i], X[:, j]))
        rff.append(scores)

    start_feature_index = np.argmax(np.array(merit_feature_class))
    selected_indices = set()
    selected_indices_list = []
    selected_indices.add(start_feature_index)
    selected_indices_list.append(start_feature_index)

    sum_cf = merit_feature_class[start_feature_index]
    sum_ff = 0
    num_ff = 0

    for _ in range(k - 1):
        current_merits = []
        num_ff += len(selected_indices)
        for i in range(num_features):
            if i not in selected_indices:
                sum_cf += merit_feature_class[i]
                added_sum_ff = 0
                for j in selected_indices:
                    if j > i:
                        added_sum_ff += rff[i][j]
                    else:
                        added_sum_ff += rff[j][i]
                sum_ff += added_sum_ff
                merit = sum_cf / math.sqrt(
                    len(selected_indices) + 1 + ((sum_ff / num_ff) * len(selected_indices) * len(selected_indices) + 1))
                current_merits.append(merit)
                sum_cf -= merit_feature_class[i]
                sum_ff -= added_sum_ff
            else:
                current_merits.append(-float('inf'))
        max_score_index = np.argmax(np.array(current_merits))
        selected_indices_list.append(max_score_index)

    merit_feature_class = np.array(merit_feature_class)
    all_scores = merit_feature_class[selected_indices_list]
    return selected_indices_list, all_scores


def mrmr_algorithm(X, y, nb_features, method):
    if method == "CFS":
        selected_indices, all_scores = CFS(X, y, nb_features)
    else:
        selected_indices, all_scores = apply_mrmr(X, y, nb_features, method)

    return selected_indices, all_scores
