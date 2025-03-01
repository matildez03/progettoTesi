#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:22:01 2025

@author: matildezoccolillo

funzione per il testing

testa il modello sul test set dato in input, 
ne stampa le metriche di performance e plotta la matrice di confusione
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Funzione per testare e stampare i risultati
def test_model(model, model_name, X_test, y_test):

    y_pred = model.predict(X_test)

    print(f"\n{model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Creazione e visualizzazione della matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Matrice di Confusione - {model_name}')
    plt.show()