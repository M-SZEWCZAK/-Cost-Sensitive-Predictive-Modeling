import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from eda.scoring_function import score_model_optimal_k
def train_model_(X_train, y_train, X_test, y_test):
    clf = XGBClassifier()