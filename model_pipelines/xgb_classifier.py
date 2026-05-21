import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import gc
from model_pipelines.auxilliary_functions import *
from eda.scoring_function import score_model_optimal_k
def train_model_growing_subset(X_train, y_train, X_test, y_test,max_subset,plot=True):
    #max subset is a list of columns to be considered in final training
    x_train_=X_train.loc[:,max_subset]
    x_test_=X_test.loc[:,max_subset]
    custom_scores=np.zeros(len(max_subset))
    precision_scores=np.zeros(len(max_subset))
    accuracy_scores=np.zeros(len(max_subset))
    recall_scores=np.zeros(len(max_subset))
    clf1=XGBClassifier()
    clf1.fit(x_train_.iloc[:,:1],y_train)
    y_pred=clf1.predict(x_test_.iloc[:,:1])
    y_proba=clf1.predict_proba(x_test_.iloc[:,0])[:,1]
    best_model=clf1
    best_subset=1
    recall_scores[0]=recall_score(y_test,y_pred)
    precision_scores[0]=precision_score(y_test,y_pred)
    accuracy_scores[0]=accuracy_score(y_test,y_pred)
    custom_scores[0]=score_model_optimal_k(y_test,y_proba,1)[0]
    for i in range(1,len(max_subset)):
        clf = XGBClassifier()
        clf.fit(x_train_.iloc[:,:i+1], y_train)
        y_pred = clf.predict(x_test_.iloc[:,:i+1])
        y_proba = clf.predict_proba(x_test_.iloc[:,:i+1])[:,1]
        recall_scores[i] = recall_score(y_test, y_pred)
        precision_scores[i] = precision_score(y_test, y_pred)
        accuracy_scores[i] = accuracy_score(y_test, y_pred)
        custom_scores[i] = score_model_optimal_k(y_test, y_proba, i+1)[0]
        if custom_scores[i]>custom_scores[best_subset-1]:
            best_subset=i+1
            best_model=clf
    features_selected=x_train_.iloc[:,:best_subset].columns.tolist()
    if plot:
        x_axis=np.arange(len(max_subset))
        plt.plot(x_axis,recall_scores,label="Recall")
        plt.plot(x_axis,precision_scores,label="Precision")
        plt.plot(x_axis,accuracy_scores,label="Accuracy")
        plt.title("Standard metrics vs subset size")
        plt.xlabel("Subset size")
        plt.legend()
        plt.show()
        plt.plot(x_axis,custom_scores,label="Custom")
        plt.title("Custom metric vs subset size")
        plt.xlabel("Subset size")
        plt.ylabel("Custom metric")
        plt.show()
    return best_model,features_selected
def train_model_all_combinations(X_train,y_train,X_test,y_test,max_subset,max_depth=1,n_estimators=3000,plot=True,return_metrics=False,test_ratio=0.2):
    possible_subsets=get_subsets(max_subset)
    n_subsets=len(possible_subsets)
    custom_scores=np.zeros(n_subsets)
    precision_scores=np.zeros(n_subsets)
    accuracy_scores=np.zeros(n_subsets)
    recall_scores=np.zeros(n_subsets)
    best_subset=0
    lens = [len(subset) for subset in possible_subsets]
    for i in range(n_subsets):
        clf = XGBClassifier(max_depth=max_depth,n_estimators=n_estimators,n_jobs=-1)
        clf.fit(X_train.loc[:, possible_subsets[i]], y_train)
        y_pred = clf.predict(X_test.loc[:, possible_subsets[i]])
        y_proba = clf.predict_proba(X_test.loc[:, possible_subsets[i]])[:, 1]
        recall_scores[i] = recall_score(y_test, y_pred)
        precision_scores[i] = precision_score(y_test, y_pred)
        accuracy_scores[i] = accuracy_score(y_test, y_pred)
        custom_scores[i] = score_model_optimal_k(y_test, y_proba, lens[i],feature_penalty=200*test_ratio,max_k=1000*test_ratio)[0]
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
        plt.show()
        plt.scatter(lens,accuracy_scores,label="Accuracy")
        plt.xlabel("Subset size")
        plt.ylabel("Accuracy")
        plt.show()
        plt.scatter(lens,recall_scores,label="Recall")
        plt.xlabel("Subset size")
        plt.ylabel("Recall")
        plt.show()
        plt.scatter(lens,custom_scores,label="Custom")
        plt.xlabel("Subset size")
        plt.ylabel("Custom score")
        plt.show()
    if return_metrics:
        df = pd.DataFrame(
            {
                "Custom Score": custom_scores,
                "Precision": precision_scores,
                "Accuracy": accuracy_scores,
                "Recall": recall_scores,
            },
        ).set_index("Subset index")
        return best_model,features_selected,df
    return best_model,features_selected
def final_xgb_hyperparameter_grid_optimizer(x_train,y_train,x_test,y_test):
    max_depths=[1,2,3,5]
    n_estimatorss=[500,1000,2000,3000]
    lr=[0.001,0.005,0.01,0.1]
    best_custom=0
    best_precision=0
    best_custom_hyperparameters={
        "max_depth":max_depths[0],
        "n_estimators":n_estimatorss[0],
        "lr":lr[0]
    }
    best_precision_hyperparameters = {
        "max_depth": max_depths[0],
        "n_estimators": n_estimatorss[0],
        "lr": lr[0]
    }
    n_features=x_train.shape[1]
    for max_depth in max_depths:
        for n_estimators in n_estimatorss:
            for j in range(len(lr)):
                clf = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators,learning_rate=lr[j],n_jobs=-1)
                clf.fit(x_train,y_train)
                y_pred = clf.predict(x_test)
                y_proba = clf.predict_proba(x_test)[:,1]
                precision=precision_score(y_test, y_pred)
                custom=score_model_optimal_k(y_test, y_proba,n_features )[0]
                if custom>best_custom:
                    best_custom_hyperparameters['max_depth']=max_depth
                    best_custom_hyperparameters['n_estimators']=n_estimators
                    best_custom_hyperparameters['lr']=lr[j]
                    best_custom=custom
                if precision>best_precision:
                    best_precision_hyperparameters['max_depth']=max_depth
                    best_precision_hyperparameters['n_estimators']=n_estimators
                    best_precision_hyperparameters['lr']=lr[j]
                    best_precision=precision
                print(f"hyperparameters: max depth={max_depth}, n_estimators={n_estimators}, lr={lr[j]}, custom_score={custom}")
    return best_custom_hyperparameters,best_precision_hyperparameters


