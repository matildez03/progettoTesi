#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:37:32 2025

@author: matildezoccolillo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Caricamento del dataset pulito
df = pd.read_csv('spotify_dataset_cleaned.csv')

# Selezioniamo un sottoinsieme del dataset per ridurre le dimensioni (es. il 30% dei dati)
df_reduced = df.sample(frac=0.3, random_state=42)

# Definiamo le feature e il target
features = ['duration_ms','explicit','mode','speechiness','instrumentalness','liveness','tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
target = 'popularity_class'

X = df_reduced[features]
y = df_reduced[target]

# Suddivisione in training (60%), validation (20%) e test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Stampa delle dimensioni dei set
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Validation Set: {X_valid.shape[0]} samples")
print(f"Test Set: {X_test.shape[0]} samples")


# Definizione della funzione per ottimizzare gli iperparametri e addestrare il modello
def train_and_tune_model(model, param_grid, X_train, y_train, X_valid, y_valid, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy', refit=True)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Valutazione sul validation set
    y_pred_valid = best_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred_valid)
    
    # Se il modello è RandomForestClassifier, calcola l'importanza delle feature
    if isinstance(model, RandomForestClassifier):
        feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        selected_features = feature_importance[:8].index.tolist()  # Prendiamo le prime 8 feature più importanti
        print(f"Feature selezionate dopo feature importance: {selected_features}")

    print(f"\n{model_name} Best Params: {best_params}")
    print(f"{model_name} Accuracy on Validation Set: {accuracy:.4f}")
    print(classification_report(y_valid, y_pred_valid))
    
    return best_model

# Definizione degli iperparametri per la ricerca
param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

param_grid_gb = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

# Training e tuning dei modelli
rf_model = train_and_tune_model(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train, X_valid, y_valid, "Random Forest")
gb_model = train_and_tune_model(GradientBoostingClassifier(random_state=42), param_grid_gb, X_train, y_train, X_valid, y_valid, "Gradient Boosting")
svm_model = train_and_tune_model(SVC(), param_grid_svm, X_train, y_train, X_valid, y_valid, "SVM")
