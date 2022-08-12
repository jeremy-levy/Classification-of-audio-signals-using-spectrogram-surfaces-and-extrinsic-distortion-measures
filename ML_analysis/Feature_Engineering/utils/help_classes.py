from enum import Enum, IntEnum


class ModelsEnum(IntEnum):
    """
    Classifier to use, for the init call of the class ML.model.Model
    """
    RF = 1
    """
    Random Forest
    """
    SVM = 2
    """
    Support Vector Machine
    """
    AB = 3
    """
    AdaBoost
    """
    LR = 4
    """
    Logistic Regression
    """
    KNN = 5
    """
    K-Nearest Neighbors
    """
    Xgboost = 6
    """
    Xgboost classifier
    """
    Catboost = 7
    """
    CatBoost classifier
    """
    LightGBM = 8
    """
    LightGBM classifier
    """


class ScalerEnum(Enum):
    """
    Prepocessing to use, for the init call of the class ML.process_data.Data
    """
    min_max = 1
    max_abs = 2
    standard = 3
    robust = 4


class NotTrained(Exception):
    pass


class __FeatureSelectionTechnique(Enum):
    RF = 1
    SHAP = 2
    Chi2 = 3
    MIS = 4
    Lasso = 5


class HyperParamSearch(Enum):
    GridSearchCV = 1
    RandomizedSearchCV = 2
    BayesSearchCV = 3
