# 📡 Telecom Churn AI

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Tests](https://img.shields.io/badge/Tests-Passing-green)
![Status](https://img.shields.io/badge/Status-POC_Completed-brightgreen)

## 📄 Contexte du Projet

Ce projet répond à un besoin critique d'une entreprise de télécommunications : **prédire le désabonnement (Churn) de ses clients**.
L'objectif est de développer un pipeline de Machine Learning supervisé capable d'identifier les clients à risque afin d'orienter les stratégies de fidélisation de l'équipe marketing.

## 🎯 Objectifs Réalisés
- **Exploration (EDA)** : Analyse des corrélations et distribution des données via Jupyter Notebook.
- **Pipeline Automatisé** : Script Python gérant le chargement, le nettoyage, l'encodage et l'entraînement.
- **Qualité Code** : Mise en place de tests unitaires (`pytest`) pour valider la robustesse du code.
- **Modélisation** : Comparaison des performances entre *Logistic Regression* et *Random Forest*.

---

## 📂 Structure du Projet

```bash
├── data/
│   └── raw/
│       └── ChurnDataFile.csv    # Données sources (Ne jamais modifier directement)
├── notebooks/
│   └── NoteBookJupyter.ipynb                # Notebook Jupyter : Exploration (EDA) et brouillon
├── src/
│   ├── pipeline.py              # Pipeline complet : Préparation, Entraînement, Évaluation
│   └── test_pipeline.py         # Tests unitaires (pytest) pour valider le pipeline
├── ChurnEnvr/                   # Environnement virtuel Python (contient les librairies)
├── requirements.txt             # Liste des dépendances (pandas, sklearn, pytest...)
└── README.md                    # Documentation du projet
```

-----

## 🚀 Installation et Lancement

### 1\. Installation de l'environnement

```bash
# Création de l'environnement virtuel
python -m venv ChurnEnvr

# Activation (Windows)
.\ChurnEnvr\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

### 2\. Exécution du Pipeline

Pour lancer le chargement des données, l'entraînement et voir les résultats :

```bash
python src/pipeline.py
```

### 3\. Exécution des Tests

Pour vérifier que tout fonctionne (Data Quality & Logic) :

```bash
pytest src/test_pipeline.py
```

*Résultat attendu : `2 passed`*

-----

## 📊 Résultats et Performance

Deux modèles ont été entraînés et comparés sur un jeu de test de **1409 clients** (20% du dataset).

| Métrique | Régression Logistique (Retenu) | Random Forest |
|----------|--------------------------------|---------------|
| **Accuracy** | **79.8%** | 79.1% |
| **Recall (Rappel)** | **54.3%** | 50.0% |
| **F1-Score** | **0.59** | 0.56 |
| **ROC AUC** | **0.84** | 0.82 |

### 🧠 Analyse Technique

Le modèle **Régression Logistique** a été sélectionné pour la mise en production.

1.  **Meilleure détection (Recall) :** Il identifie mieux les clients qui vont réellement partir (54.3%) comparé au Random Forest.
2.  **Robustesse (ROC AUC) :** Avec un score de 0.84, il offre une excellente capacité de discrimination entre les clients fidèles et les désabonnés.
3.  **Simplicité :** Modèle plus léger et plus rapide à interpréter.

-----

## ⚙️ Détails du Pipeline (Feature Engineering)

Le script `pipeline.py` effectue automatiquement les transformations suivantes :

1.  **Nettoyage** : Gestion des valeurs vides dans `TotalCharges`.
2.  **Encodage** : Transformation des variables catégorielles (ex: 'Yes'/'No' -\> 1/0) via `LabelEncoder`.
3.  **Normalisation** : Mise à l'échelle des variables numériques via `MinMaxScaler` pour optimiser la convergence des algorithmes.
4.  **Split Stratifié** : Division Train/Test respectant la proportion de Churn initial.


