import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from eda.scoring_function import score_model_optimal_k
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
"""
The functions provided here are supposed to be first level elimination, so that only a handful 
of best features are considered in final checks, where the remaining features will be evaluated in
many configurations. 
"""


__all__ = ['reduce_multicollinearity','remove_highly_correlated','select_best_mutual_information','select_best_correlation','Kolmogorov_Smirnov_selector',
           'random_forest_selector','xgb_selector','boruta_handler','SHAP_selector']
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
    y_train = y_train.values.ravel()
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
# === selectors based on tree-based models, with selectable model type
def boruta_handler(x,y,max_depth=3,model_type='rf',return_='support'):
    if return_ not in ['support','ranking','all']:
        raise ValueError("return_ must be one of ['support','ranking','all']")
    if model_type not in ['rf','xgb']:
        raise ValueError('model_type must be either "rf" or "xgb"')
    if model_type == 'xgb':
        model=XGBClassifier(class_weight='balanced',max_depth=max_depth)
    else:
        model=RandomForestClassifier(class_weight='balanced',max_depth=max_depth)
    x_n=x.values
    y_n=y.values.ravel()
    feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(x_n, y_n)
    print(feat_selector.support_)
    print(feat_selector.ranking_)
    if return_ == 'support':
        return feat_selector.support_
    elif return_ == 'ranking':
        return feat_selector.ranking_
    else:
        return feat_selector.support_,feat_selector.ranking_

def SHAP_selector(x_train,y_train,n_features,modeltype='xgb',n_estimators=1000,max_depth=3,present_shap_plot=True,random_state=2137):
    if modeltype not in ['xgb','rf']:
        raise ValueError('modeltype must be either "xgb" or "rf"')
    if modeltype == 'xgb':
        model=XGBClassifier(class_weight='balanced',max_depth=max_depth,n_estimators=n_estimators,learning_rate=0.01,n_jobs=-1,random_state=random_state)
    else:
        model=RandomForestClassifier(class_weight='balanced',max_depth=max_depth,n_estimators=n_estimators,random_state=random_state)
    model.fit(x_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_train)
    if len(shap_values.values.shape) == 3:
        raw_shap_matrix = shap_values.values[:, :, 1]
    else:
        raw_shap_matrix = shap_values.values
    if present_shap_plot:
        plt.figure(figsize=(10, 5))
        plot_obj = shap.Explanation(
            values=raw_shap_matrix,
            data=x_train.values,
            feature_names=x_train.columns
        )
        shap.plots.bar(plot_obj, max_display=10)
        plt.title("Global Feature Importance via SHAP")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(plot_obj, max_display=10)
        plt.title("SHAP Beeswarm Plot: Feature Value vs. Model Impact")
        plt.tight_layout()
        plt.show()
    mean_abs_shap = np.mean(np.abs(raw_shap_matrix), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_features = x_train.columns[sorted_indices]
    return sorted_features[:n_features].tolist()


# ===== model specific selectors ====
def random_forest_selector(x_train,y_train,n_features,vif_check=False,n_estimators=1000,max_depth=3,random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, random_state=random_state)
    if vif_check:
        x_train=reduce_multicollinearity(x_train,5)
    model.fit(x_train, y_train)
    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    importances = importances.sort_values(ascending=False)
    return importances.head(n_features).index.tolist()
def xgb_selector(x_train,y_train,n_features,importance_type="gain",vif_check=False,n_estimators=1000,max_depth=3,learning_rate=0.01,random_state=42,return_='weight_df'):
    model= XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate, random_state=random_state)
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
        if return_ == "weight_df":
            return gain_importance_df.head(n_features)
        else:
            return gain_importance_df.head(n_features).loc[:,'Feature'].tolist()
    elif importance_type == "weight":
        if return_ == "weight_df":
            return weight_importance_df.head(n_features)
        else:
            return weight_importance_df.head(n_features).loc[:,'Feature'].tolist()
    else:
        raise ValueError("Available importance methods are \"gain\" and \"weight\"")









