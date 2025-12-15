import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, roc_auc_score

def load_data(filepath="./data/raw/ChurnDataFile.csv"):
    """Charge les données depuis un fichier CSV."""
    # Permet de gérer les chemins relatifs que l'on lance depuis src ou la racine
    if not os.path.exists(filepath):
        # Tentative de remonter d'un cran si lancé depuis src/
        filepath = os.path.join("..", filepath)
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"✅ Données chargées : {df.shape}")
    return df

def preprocess_data(df):
    """Nettoie, encode et normalise les données."""
    df = df.copy() # Pour ne pas modifier l'original
    
    # 1. Conversion TotalCharges (Force numérique)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # 2. Suppression des colonnes inutiles (ID)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # 3. Encodage binaire de la cible (Churn)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # 4. Encodage des variables catégorielles (LabelEncoder)
    # Note: En production réelle, on préfère OneHotEncoder ou sauvegarder le LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # 5. Normalisation (MinMax) sur les colonnes numériques (sauf la cible)
    target = None
    if 'Churn' in df.columns:
        target = df['Churn']
        df = df.drop(columns=['Churn'])
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # On remet la cible si elle existait
    if target is not None:
        df_scaled['Churn'] = target.reset_index(drop=True)
        
    return df_scaled

def split_data(df, target_col='Churn', test_size=0.2):
    """Divise le dataset en Train et Test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Entraîne plusieurs modèles."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"🤖 Modèle entraîné : {name}")
        
    return trained_models

def evaluate_model(model, X_test, y_test):
    """Évalue un modèle et retourne les métriques."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    
    # Gestion du ROC AUC qui a besoin de probabilités
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except:
        metrics["roc_auc"] = "N/A"
        
    return metrics

# ==========================================
# Zone d'exécution principale (Main)
# ==========================================
if __name__ == "__main__":
    # Cette partie ne s'exécute QUE si tu lances "python pipeline.py"
    # Elle ne s'exécute PAS si tu fais "import pipeline"
    
    try:
        # 1. Chargement
        df = load_data()
        
        # 2. Préparation
        df_clean = preprocess_data(df)
        
        # 3. Split
        X_train, X_test, y_train, y_test = split_data(df_clean)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 4. Entraînement
        models = train_models(X_train, y_train)
        
        # 5. Évaluation et Choix
        print("\n📊 RÉSULTATS :")
        for name, model in models.items():
            res = evaluate_model(model, X_test, y_test)
            print(f"--- {name} ---")
            print(res)
            
    except Exception as e:
        print(f"❌ Erreur : {e}")