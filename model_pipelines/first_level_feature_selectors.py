import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
"""
The functions provided here are supposed to be first level elimination, so that only a handful 
of best features are considered in final checks, where the remaining features will be evaluated in
many configurations. 
"""


__all__ = ['reduce_multicollinearity','remove_highly_correlated','select_best_mutual_information','select_best_correlation','Kolmogorov_Smirnov_selector',
           'random_forest_selector','xgb_selector']
# ===== collinearity reduction =======


def reduce_multicollinearity(df, threshold=5.0):
    """
    Iteratively removes features with a VIF above a specified threshold. For a big number of features
    it is too slow.

    Parameters:
    df (DataFrame): The input dataframe containing only the independent numerical features.
    threshold (float): The VIF threshold. Features with a VIF higher than this will be dropped.
                       Common thresholds are 5.0 or 10.0.

    Returns:
    DataFrame: A new dataframe with the highly correlated features removed.
    """
    X = df.copy()

    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break
        max_vif_feature = vif_data.sort_values(by="VIF", ascending=False).iloc[0]["feature"]
        print(f"Dropping '{max_vif_feature}' (VIF: {max_vif:.2f})")

        X = X.drop(columns=[max_vif_feature])
        if X.shape[1] <= 1:
            break

    print("\nFeature reduction complete.")
    return X
def remove_highly_correlated(df,threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    df_dropped = df.drop(columns=to_drop)
    print(f"Original features count: {df.shape[1]}")
    print(f"Dropped features count:  {len(to_drop)}")
    print(f"Remaining features count: {df_dropped.shape[1]}")
    return df_dropped
# ===== model-agnostic selectors =======
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
def boruta_handler(x,y,max_depth=3,model_type='rf',return_='filtered_df'):
    if return_ not in ['filtered_df','support','ranking','all']:
        raise ValueError("return_ must be one of ['filtered_df','support','ranking','all']")
    if model_type not in ['rf','xgb']:
        raise ValueError('model_type must be either "rf" or "xgb"')
    if model_type == 'xgb':
        model=XGBClassifier(class_weight='balanced',max_depth=max_depth)
    else:
        model=RandomForestClassifier(class_weight='balanced',max_depth=max_depth)
    x_n=x.values
    y_n=y.ravel()
    feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(x_n, y_n)
    print(feat_selector.support_)
    print(feat_selector.ranking_)
    if return_ == 'filtered_df':
        pass
    elif return_ == 'support':
        return feat_selector.support_
    elif return_ == 'ranking':
        return feat_selector.ranking_
    else:
        return feat_selector.support_,feat_selector.ranking_



# ===== model specific selectors ====
def random_forest_selector(x_train,y_train,n_features,vif_check=False):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    if vif_check:
        x_train=reduce_multicollinearity(x_train,5)
    model.fit(x_train, y_train)
    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    importances = importances.sort_values(ascending=False)
    return importances.head(n_features).index.tolist()
def xgb_selector(x_train,y_train,n_features,importance_type="gain",vif_check=False):
    model= XGBClassifier(n_estimators=100, random_state=42)
    if vif_check:
        x_train=reduce_multicollinearity(x_train,5)
    model.fit(x_train, y_train)
    booster = model.get_booster()
    gain_importance = booster.get_score(importance_type='gain')
    weight_importance = booster.get_score(importance_type='weight')
    gain_importance_df = pd.DataFrame({
        'Feature': gain_importance.keys(),
        'Gain (Importance)': gain_importance.values()
    }).sort_values(by='Gain (Importance)', ascending=False)
    weight_importance_df = pd.DataFrame({
        'Feature': gain_importance.keys(),
        'Weight (Importance)': weight_importance.values()
    }).sort_values(by='Weight (Importance)', ascending=False)
    if importance_type == "gain":
        return gain_importance_df.head(n_features)
    elif importance_type == "weight":
        return weight_importance_df.head(n_features)
    else:
        raise ValueError("Available importance methods are \"gain\" and \"weight\"")









