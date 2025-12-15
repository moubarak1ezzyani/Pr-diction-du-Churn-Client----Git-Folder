import pytest
import pandas as pd
import numpy as np
from src.pipeline import preprocess_data, split_data

# Fixture : Crée des fausses données pour les tests
@pytest.fixture
def dummy_data():
    # On multiplie les listes par 4 pour avoir 20 lignes au lieu de 5
    # Cela permet au train_test_split d'avoir assez de données pour stratifier
    data = {
        'customerID': [str(i) for i in range(20)],
        'TotalCharges': ['100', '200', ' ', '400', '500'] * 4, 
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'] * 4,
        'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes'] * 4
    }
    return pd.DataFrame(data)

def test_preprocess_data(dummy_data):
    """Vérifie que le nettoyage fonctionne (plus de NaN, conversion numérique)"""
    df_clean = preprocess_data(dummy_data)
    
    # Vérifier qu'il n'y a plus de CustomerID
    assert 'customerID' not in df_clean.columns
    
    # Vérifier que Churn est bien 0 ou 1
    assert df_clean['Churn'].isin([0, 1]).all()
    
    # Vérifier qu'il n'y a plus de valeurs manquantes
    assert df_clean.isnull().sum().sum() == 0
    
    # Vérifier que TotalCharges est bien numérique
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])

def test_split_dimensions(dummy_data):
    """Vérifie que le split respecte les dimensions"""
    df_clean = preprocess_data(dummy_data)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
    
    # Vérifier la cohérence X et Y
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Vérifier que la somme fait bien le total
    assert len(X_train) + len(X_test) == len(df_clean)