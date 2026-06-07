import gc
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from eda.scoring_function import *
from sklearn.model_selection import train_test_split

from model_pipelines.auxilliary_functions import get_subsets


def train_model_all_combinations(X_train,y_train,X_test,y_test,max_subset,max_depth=1,n_estimators=3000,plot=True,save_plot=False,save_path=None,return_metrics=False,test_ratio=0.2):
    if save_plot and save_path is None:
        print("No path to save plot provided! Plots will not be saved!")
        save_plot = False
    elif save_plot and save_path is not None:
        dir_path = Path(save_path)
        prec_path= dir_path / "precision.png"
        rec_path= dir_path / "recall.png"
        acc_path= dir_path / "accuracy.png"
        cust_path= dir_path / "custom_score.png"
    possible_subsets=get_subsets(max_subset)
    n_subsets=len(possible_subsets)
    custom_scores=np.zeros(n_subsets)
    precision_scores=np.zeros(n_subsets)
    accuracy_scores=np.zeros(n_subsets)
    recall_scores=np.zeros(n_subsets)
    best_subset=0
    lens = [len(subset) for subset in possible_subsets]
    for i in range(n_subsets):
        clf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,n_jobs=-1,random_state=42)
        clf.fit(X_train.loc[:, possible_subsets[i]], y_train)
        y_proba = clf.predict_proba(X_test.loc[:, possible_subsets[i]])[:, 1]
        custom_scores[i],best_thr = score_model_optimal_k(y_test, y_proba, lens[i],feature_penalty=200*test_ratio,max_k=int(1000*test_ratio))[:2]
        y_pred=(y_proba>best_thr).astype(int)
        recall_scores[i] = recall_score(y_test, y_pred)
        precision_scores[i] = precision_score(y_test, y_pred)
        accuracy_scores[i] = accuracy_score(y_test, y_pred)
        if i==0:
            best_model=clf
        elif custom_scores[i] > custom_scores[best_subset]:
            best_subset = i
            best_model = clf
        gc.collect()
    features_selected=possible_subsets[best_subset]
    if plot:
        plt.scatter(lens,precision_scores,label="Precision")
        plt.xlabel("Subset size")
        plt.ylabel("Precision")
        if save_plot:
            plt.savefig(prec_path)
        plt.show()
        plt.scatter(lens,accuracy_scores,label="Accuracy")
        plt.xlabel("Subset size")
        plt.ylabel("Accuracy")
        if save_plot:
            plt.savefig(acc_path)
        plt.show()
        plt.scatter(lens,recall_scores,label="Recall")
        plt.xlabel("Subset size")
        plt.ylabel("Recall")
        if save_plot:
            plt.savefig(rec_path)
        plt.show()
        plt.scatter(lens,custom_scores,label="Custom")
        plt.xlabel("Subset size")
        plt.ylabel("Custom score")
        if save_plot:
            plt.savefig(cust_path)
        plt.show()
    if return_metrics:
        df = pd.DataFrame(
            {
                "Subset index": range(len(custom_scores)),
                "Custom Score": custom_scores,
                "Precision": precision_scores,
                "Accuracy": accuracy_scores,
                "Recall": recall_scores,
            },
        ).set_index("Subset index")
        subset_dict={
            i:possible_subsets[i]
            for i in range(len(custom_scores))
        }
        return best_model,features_selected,df,subset_dict
    return best_model,features_selected
def final_rf_hyperparameter_grid_optimizer(x_train,y_train,x_test,y_test,test_ratio=0.2):
    max_depths=[1,2,3]
    n_estimatorss=[500,1000,2000,3000]
    best_custom=0
    best_precision=0
    best_custom_hyperparameters={
        "max_depth":max_depths[0],
        "n_estimators":n_estimatorss[0],
    }
    best_precision_hyperparameters = {
        "max_depth": max_depths[0],
        "n_estimators": n_estimatorss[0],
    }
    n_features=x_train.shape[1]
    for max_depth in max_depths:
        for n_estimators in n_estimatorss:
                clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,n_jobs=-1,random_state=42)
                clf.fit(x_train,y_train)
                y_proba = clf.predict_proba(x_test)[:,1]
                custom,best_thr=score_model_optimal_k(y_test, y_proba,n_features,max_k=int(1000*test_ratio),feature_penalty=200*test_ratio )[:2]
                y_pred = (y_proba > best_thr).astype(int)
                precision=precision_score(y_test, y_pred)
                if custom>best_custom:
                    best_custom_hyperparameters['max_depth']=max_depth
                    best_custom_hyperparameters['n_estimators']=n_estimators
                    best_custom=custom
                if precision>best_precision:
                    best_precision_hyperparameters['max_depth']=max_depth
                    best_precision_hyperparameters['n_estimators']=n_estimators
                    best_precision=precision
                # print(f"hyperparameters: max depth={max_depth}, n_estimators={n_estimators}, lr={lr[j]}, custom_score={custom}")
    return best_custom_hyperparameters,best_precision_hyperparameters
