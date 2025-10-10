from .pipeline import load_data, Prepare_To_Training,Training_Model, comp_models, encode_Category  
import pandas as pds
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pytest

def test_dimensions_split(Data_File):
    X_train, X_test, Y_train, Y_test = Prepare_To_Training(Data_File)
    # Vérifier la cohérence des dimensions
    assert len(X_train) == len(Y_train) # X_train et y_train n'ont pas le même nombre de lignes
    assert len(X_test) == len(Y_test) # X_test et y_test n'ont pas le même nombre de lignes

print("Les dimensions sont coherentess")
# -----> Absence de valeurs manquantes

# -----> Dimensions cohérentes entre X et y après split
# -----> Types de colonnes corrects après transformation