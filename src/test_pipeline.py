import pipeline 
import pandas as pds
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier 

def Training_Model():   
    model = DecisionTreeClassifier(random_state=42)
    X_train, X_test, Y_train, Y_test= model.fit(X_train, Y_train)
    Y_prediction =model.predict(X_test)
    Accuracy = accuracy_score(Y_test, Y_prediction)
    print(f"\nPrécision (Accuracy) sur l'ensemble de test: {Accuracy:.2f}")
    print("\nRapport de classification:\n", classification_report(Y_test, Y_prediction))
# Prepare_To_Training()
print("===== Check Training =====")
print(Training_Model())
# -----> Absence de valeurs manquantes

# -----> Dimensions cohérentes entre X et y après split
# -----> Types de colonnes corrects après transformation