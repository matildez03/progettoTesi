#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:07:08 2025

@author: matildezoccolillo

selezione degli iperparametri
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def optimize_hyperparameters(model, param_grid, X_train, y_train, sample_size=0.3, cv=6):
    """
    Esegue GridSearchCV su un sottoinsieme dei dati per selezionare i migliori iperparametri.
    
    :param model: Il modello ML da ottimizzare
    :param param_grid: Dizionario dei parametri da testare
    :param X_train: Feature di training complete
    :param y_train: Target di training completo
    :param sample_size: Percentuale di dati da usare per la ricerca (default: 30%)
    :param cv: Numero di fold per la cross-validation (default: 6)
    :return: Il modello ottimizzato con i migliori iperparametri
    """
    
    # Campiona un sottoinsieme del training set
    X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=(1 - sample_size), stratify=y_train, random_state=42)
    
    # Esegue GridSearchCV sul sottoinsieme
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_sample, y_sample)
    
    # Recupera il modello con i migliori iperparametri
    best_model = grid_search.best_estimator_
    
    print(f"\nMigliori parametri trovati per {model.__class__.__name__}: {grid_search.best_params_}")
    
    # Addestra il modello ottimizzato su tutto il dataset di training
    best_model.fit(X_train, y_train)
    
    return best_model
