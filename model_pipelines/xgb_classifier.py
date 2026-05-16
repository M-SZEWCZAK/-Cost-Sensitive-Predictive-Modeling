import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier

from eda.scoring_function import score_model_optimal_k
def train_model_growing_subset(X_train, y_train, X_test, y_test,max_subset,plot=True):
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
def train_model_all_combinations(X_train,y_train,X_test,y_test,max_subset,plot=True):
    pass
