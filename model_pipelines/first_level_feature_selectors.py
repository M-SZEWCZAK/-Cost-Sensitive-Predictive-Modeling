import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from eda.scoring_function import score_model_optimal_k
from sklearn.feature_selection import mutual_info_classif

def select_best_mutual_information(x_train,y_train,n_features):
    """
    :param x_train: training data
    :param y_train: training labels
    :param n_features: number of features to be selected
    :return: a list of n_features features with best mutual information with target value
    """
    mi_scores=mutual_info_classif(x_train,y_train.values.ravel())
    mi_series = pd.Series(mi_scores, name="MI Scores", index=x_train.columns)
    mi_series = mi_series.sort_values(ascending=True)
    non_zero_features = mi_series[mi_series > 0].index.tolist()
    if len(non_zero_features) >n_features:
        return non_zero_features[:n_features]
    else:
        print(f"only {len(non_zero_features)} features selected. other features have "
              f"mutual information equal zero with target value")
        return non_zero_features
def select_best_correlation(x_train,y_train,n_features):
    """
  :param x_train: training data
    :param y_train: training labels
    :param n_features: number of features to be selected
    :return: a list of n_features features with best absolute value of correlation with target value
    """
    train = pd.concat([x_train, y_train], axis=1)
    corr_matrix = train.corr()
    target_col_name = train.columns[-1]
    target_corr = corr_matrix[target_col_name]
    target_corr = target_corr.drop(labels=[target_col_name])
    abs_target_corr = abs(target_corr)
    best_features = abs_target_corr.sort_values(ascending=False).head(n_features).index.tolist()
    return best_features
def Kolmogorov_Smirnov_selector(x_train,y_train,n_features):
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train)
    feature_cols = x_train.select_dtypes(include=['number']).columns
    classes = pd.Series(y_train).unique()
    if len(classes) != 2:
        raise ValueError("This method is designed strictly for binary classification (2 classes).")
    mask_0 = (y_train == classes[0])
    mask_1 = (y_train == classes[1])
    ks_results = []
    for col in feature_cols:
        group1 = x_train.loc[mask_0, col].dropna()
        group2 = x_train.loc[mask_1, col].dropna()
        stat, _ = ks_2samp(group1, group2)
        ks_results.append({
            'feature': col,
            'ks_statistic': stat
        })
    ranking_df = pd.DataFrame(ks_results).sort_values(by='ks_statistic', ascending=False)
    top_n_features = ranking_df['feature'].head(n_features).tolist()
    return top_n_features, ranking_df





