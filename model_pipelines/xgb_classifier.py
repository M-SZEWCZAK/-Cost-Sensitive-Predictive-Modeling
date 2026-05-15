import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from eda.scoring_function import score_model_optimal_k
def train_model_growing_subset(X_train, y_train, X_test, y_test,max_subset):
    x_train_=X_train.loc[max_subset]
    x_test_=X_test.loc[max_subset]

    clf = XGBClassifier()
    clf.fit(X_train, y_train)