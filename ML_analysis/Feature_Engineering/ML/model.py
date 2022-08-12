import numpy as np
import pandas as pd
from sklearn import neighbors, svm, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, jaccard_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import pickle
from skopt import BayesSearchCV

from ML_analysis.Feature_Engineering.utils.help_classes import ModelsEnum, NotTrained, HyperParamSearch


class Model:

    def __init__(self, model_name: str, classifier_method: ModelsEnum, verbose: bool,
                 binary_task: bool, class_weight: dict = None, write_csv: bool = True, random_grid: dict = None,
                 GPU_catboost: bool = False):
        """

        :param model_name: Name of the model. Will be used for csv files created by the class.
        :param classifier_method: The classifier to use. See utils.ModelsEnum for different options.
        :param random_grid: Random grid for the cross-fold validation. If None is set, a default random grid will be attributed.
        :param verbose: Whether to print results or not.
        :param binary_task: Whether it is a binary task or not.
        :param class_weight: Dict with the weight of each class.
        :param write_csv: Whether to write the results to a csv file.
        :param GPU_catboost: whether to run on GPU or not, only for CatBoost
        """
        self.random_state = 32
        self.classifier_method = classifier_method
        self.classifier = self.__get_classifier(class_weight)
        self.model_name = model_name
        self.verbose = verbose
        self.trained_model = None
        self.binary_task = binary_task
        self.write_csv = write_csv
        self.GPU_catboost = GPU_catboost
        if random_grid is None:
            self.random_grid = self.__get_grid_search()
        else:
            self.random_grid = random_grid

    def __get_classifier(self, class_weight: dict):
        if self.classifier_method == ModelsEnum.AB:
            return AdaBoostClassifier(algorithm="SAMME", random_state=self.random_state)
        elif self.classifier_method == ModelsEnum.KNN:
            return neighbors.KNeighborsClassifier()
        elif self.classifier_method == ModelsEnum.LR:
            return LogisticRegression(random_state=self.random_state, max_iter=5000, verbose=0, dual=False,
                                      class_weight=class_weight)
        elif self.classifier_method == ModelsEnum.RF:
            return RandomForestClassifier(random_state=self.random_state, class_weight=class_weight)
        elif self.classifier_method == ModelsEnum.SVM:
            return svm.SVC(probability=False, random_state=self.random_state, class_weight=class_weight)
        elif self.classifier_method == ModelsEnum.Xgboost:
            return xgb.XGBClassifier(use_label_encoder=False, random_state=self.random_state)
        elif self.classifier_method == ModelsEnum.Catboost:
            # if self.GPU_catboost is True:
            #     return CatBoostClassifier(iterations=1000, task_type="GPU")
            # else:
            return CatBoostClassifier(iterations=1000)
        elif self.classifier_method == ModelsEnum.LightGBM:
            return lgb.LGBMClassifier()

    def __get_grid_search(self):
        grid_search = None

        if self.classifier_method == ModelsEnum.AB:
            n_estimators = [40, 60, 90, 100, 110, 120, 130, 140, 150, 200, 210, 220, 250, 300]
            learning_rate = np.logspace(-4, 1.2, num=10)
            base_estimator = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=5),
                              DecisionTreeClassifier(max_depth=10), svm.SVC(probability=True),
                              LogisticRegression(random_state=0, max_iter=5000, verbose=0)]

            grid_search = {'n_estimators': n_estimators,
                           'learning_rate': learning_rate,
                           "base_estimator": base_estimator,
                           }
        elif self.classifier_method == ModelsEnum.KNN:
            weights = ['uniform', 'distance']
            algorithm = ['ball_tree', 'kd_tree', 'brute']
            n_neighbors = [3, 5, 7, 11, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151]
            p = [1, 2]

            grid_search = {'n_neighbors': n_neighbors,
                           'weights': weights,
                           'algorithm': algorithm,
                           'p': p}
        elif self.classifier_method == ModelsEnum.LR:
            penalty = ['l2', 'l1', 'elasticnet']
            solver = ['newton-cg', 'lbfgs', 'sag', 'saga']
            tol = np.logspace(-9, 2, num=10)
            C = np.logspace(-9, 13, num=10)

            grid_search = {'penalty': penalty,
                           'tol': tol,
                           'C': C,
                           'solver': solver}
        elif self.classifier_method == ModelsEnum.RF:
            max_features = ['sqrt', 'log2']
            min_samples_split = [2, 5, 10, 15, ]
            min_samples_leaf = [1, 2, 4, 5, 10]
            bootstrap = [True, False]
            criterion = ['gini', 'entropy']
            n_estimators = [40, 90, 100, 110, 120, 130, 140, 150, 200, 210, 220, 250]
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)

            grid_search = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap,
                           'criterion': criterion,
                           }
        elif self.classifier_method == ModelsEnum.SVM:
            kernel = ["linear", "poly", "rbf", "sigmoid"]
            shrinking = [True, False]
            C = np.logspace(-6, 2, num=10)
            gamma = ['scale', 'auto']
            tol = np.logspace(-6, 2, num=10)
            grid_search = {'C': C,
                           'kernel': kernel,
                           'gamma': gamma,
                           'shrinking': shrinking,
                           'tol': tol,
                           }

        elif self.classifier_method == ModelsEnum.Xgboost:
            grid_search = {"n_estimators": np.linspace(10, 150, 10, dtype=int),
                           "max_depth": np.linspace(2, 15, 10, dtype=int),
                           "learning_rate": np.logspace(-5, 0, 15),
                           "reg_alpha": np.logspace(-5, 0, 5),
                           "reg_lambda": np.logspace(-5, 0, 5),
                           'min_child_weight': [1, 5, 10],
                           'gamma': [0.5, 1, 1.5, 2, 5],
                           'subsample': [0.6, 0.8, 1.0],
                           'colsample_bytree': [0.6, 0.8, 1.0],
                           }

        elif self.classifier_method == ModelsEnum.Catboost:
            grid_search = {"l2_leaf_reg": np.logspace(-5, 1, 15),
                           "depth": np.linspace(2, 15, 10, dtype=int),
                           "learning_rate": np.logspace(-5, 0, 15),
                           }

        elif self.classifier_method == ModelsEnum.LightGBM:
            grid_search = {
                'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
                'num_leaves': np.linspace(5, 60, 30, dtype=int),
                "max_depth": np.linspace(2, 15, 10, dtype=int),
                'learning_rate': np.logspace(-4, 1.2, num=10),
                "n_estimators": np.linspace(10, 150, 10, dtype=int),
                "reg_alpha": np.logspace(-5, 0, 5),
                "reg_lambda": np.logspace(-5, 0, 5),
                "bagging_freq": [1],
            }
        return grid_search

    def cross_fold_validation(self, X_train, y_train, grid_search: HyperParamSearch, cv: int = 3, scoring=None,
                              n_iter: int = 200, n_jobs: int = 1):
        """
        Perform cross fold validation. The model will be trained, and the method test_set can be called after that.

        :param X_train: Data to train on. shape: n_samples x n_features
        :param y_train: Label of the data: shape n_samples x n_classes
        :param grid_search: Bool. If true, a grid search will be performed, otherwise a randomized grid search
        :param cv: Number of iterations for cross fold validation
        :param scoring: Metric for compare each set of hyper-parameters. If None, the accuracy will be used.
        :param n_iter: Number of possibilities tried, in the case of grid_search=False
        :param n_jobs: Number of jobs for the cross-fold validation.
        :return: dict with the best parameters
        """
        if scoring is None:
            scoring = make_scorer(metrics.accuracy_score)
        if grid_search == HyperParamSearch.GridSearchCV:
            random_model = GridSearchCV(estimator=self.classifier, param_grid=self.random_grid, verbose=self.verbose,
                                        n_jobs=n_jobs, return_train_score=True, cv=cv, scoring=scoring)
        elif grid_search == HyperParamSearch.RandomizedSearchCV:
            random_model = RandomizedSearchCV(estimator=self.classifier, param_distributions=self.random_grid,
                                              n_iter=n_iter, verbose=self.verbose, random_state=self.random_state,
                                              n_jobs=n_jobs, return_train_score=True, cv=cv, scoring=scoring)
        elif grid_search == HyperParamSearch.BayesSearchCV:
            random_model = BayesSearchCV(estimator=self.classifier, search_spaces=self.random_grid, n_iter=n_iter,
                                         scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=self.verbose)
        else:
            random_model = RandomizedSearchCV(estimator=self.classifier, param_distributions=self.random_grid,
                                              n_iter=n_iter, verbose=self.verbose, random_state=self.random_state,
                                              n_jobs=n_jobs, return_train_score=True, cv=cv, scoring=scoring)

        if self.classifier_method == ModelsEnum.Xgboost:
            random_model.fit(X_train, y_train, eval_metric='logloss')
        else:
            random_model.fit(X_train, y_train)

        if self.write_csv is True:
            cross_fold_results = pd.DataFrame.from_dict(random_model.cv_results_)
            cross_fold_results.to_csv(self.model_name + "_cross_fold_results.csv", index=False)

        self.trained_model = random_model
        return self.trained_model, pd.DataFrame.from_dict(random_model.cv_results_)

    def test_set(self, X_test, y_test) -> dict:
        """
        Apply the model trained on the test.

        :param X_test:  Data to test on. shape: n_samples x n_features
        :param y_test:  Label of the data: shape n_samples x n_classes
        :return: dict with all the metrics. Also writes it into a csv file, in case write_csv was set to True during the init call
        """
        if self.trained_model is None:
            raise NotTrained("need to call to cross_fold_validation first")

        y_pred = self.trained_model.predict(X_test)

        results_dict = {
            "accuracy": [metrics.accuracy_score(y_test, y_pred)],
            "f1_": [f1_score(y_test, y_pred, average='macro')],
            "jaccard": [jaccard_score(y_test, y_pred, average="micro")],
            "recall": [recall_score(y_test, y_pred, average="micro")],
        }

        if self.binary_task is True:
            TN, FP, FN, TP = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            try:
                results_dict["roc"] = roc_auc_score(y_test, y_pred)
            except:
                if self.verbose is True:
                    print("ROC skipped, because only one class present in y_true")
            results_dict["sensitivity"] = TP / (TP + FN)
            results_dict["specificity"] = TN / (TN + FP)
            results_dict["NPV"] = TN / (TN + FN)
            results_dict["PPV"] = TP / (TP + FP)

        if self.verbose is True:
            print(results_dict)

        if self.write_csv is True:
            test_set_results = pd.DataFrame.from_dict(results_dict)
            test_set_results.to_csv(self.model_name + "_test_set_results.csv", index=False)

        return results_dict

    def plot_feature_importance(self, X, y, list_features, ax, list_colors=None, nb_features=30):
        """
        Plot feature importance. Need to be called after cross_fold_validation

        :param X: Data from which the feature importance will be based. Shape n_samples x n_features
        :param y: Label of the data. Shape: n_samples x n_classes
        :param list_features: List of string with the features used by the model
        :param ax: matplotlib ax to draw on
        :param list_colors: list of color, each color correspond to a feature. Optional
        :param nb_features: Number of features to plot. Default is 30.
        """

        if self.trained_model is None:
            raise NotTrained("need to call to cross_fold_validation first")
        assert len(list_colors) == len(list_features)

        best_params_ = self.trained_model.best_params_
        RF_model = RandomForestClassifier(**best_params_)
        RF_model.fit(X, y)

        importance = RF_model.feature_importances_
        indices = np.argsort(importance)
        scores = importance[indices]
        list_features = np.array(list_features)
        features = list_features[indices]
        list_colors = np.array(list_colors)
        list_colors = list_colors[indices]

        features = features[-nb_features:]
        scores = scores[-nb_features:]
        list_colors = list_colors[-nb_features:]

        ax.barh(features, scores, color=list_colors)

    def save_model(self):
        pickle.dump(self.trained_model, open(self.model_name, 'wb'))
