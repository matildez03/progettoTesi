#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:55:20 2025

@author: matildezoccolillo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

df = pd.read_csv("top10s.csv", encoding='ISO-8859-1') #CARICO IL Dataset
print(df.head()) #stampo la prima parte per vedere com'è strutturato il dataset
#noto che c'è una colonna senza nome
print(df.info()) #stampo le informazioni relative al dataset
#noto che non ci sono Nan e che le variabili sono tutte di tipo numerico tranne titolo, artista e genere 

#CLEANING E PREPROCESSING
df.drop(['Unnamed: 0'], axis = 1, inplace = True) #rimuovo la colonna senza nome


df["pop"].hist()
plt.show()

# CREAZIONE DELLA VARIABILE TARGET

# Creo 4 categorie bilanciate usando i quantili
df['popularity_class'] = pd.qcut(df['pop'], q=4, labels=[1, 2, 3, 4])

# Stampa il numero di istanze in ogni classe
print(df['popularity_class'].value_counts())

# Stampa il range di popularity per ogni categoria
range_by_class = df.groupby('popularity_class')['pop'].agg(['min', 'max'])
print(range_by_class)

print(df.info())

# 2 - DIVISIONE IN TRAINING E TEST SET
# Divisione 80% training, 20% test
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['popularity_class'])

# Controllo delle dimensioni
print(f"Dimensione Training Set: {df_train.shape[0]} entries")
print(f"Dimensione Test Set: {df_test.shape[0]} entries")

# CORRELAZIONE TRA FEATURES NUMERICHE E POPOLARITA'
X_num = df_train.drop(columns=["title","artist","top genre"])
plt.figure(figsize=(12,8))
sns.heatmap(X_num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione tra Feature Musicali e Popolarità")
plt.show()

#TRAINING AND TESTING
# Selezione delle feature e target
X_train = df_train.drop(columns=["title", "artist", "top genre", "pop", "popularity_class"])
X_test = df_test.drop(columns=["title", "artist", "top genre", "pop", "popularity_class"])
y_train = df_train["popularity_class"].astype(int)
y_test = df_test["popularity_class"].astype(int)

# Standardizzazione per SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelli da testare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, random_state=42)
}

# Training e valutazione
for name, model in models.items():
    print(f"\n{name}")
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Importanza delle feature per modelli basati su alberi
    if name in ["Random Forest", "Gradient Boosting"]:
        importance = model.feature_importances_
        feature_importance = sorted(zip(X_train.columns, importance), key=lambda x: x[1], reverse=True)
        print("Top 5 Feature Importances:")
        for feat, imp in feature_importance[:5]:
            print(f"{feat}: {imp:.4f}")
    
    # Importanza delle feature per SVM
    elif name == "SVM":
        result = permutation_importance(model, X_test_scaled, y_test, scoring="accuracy", random_state=42)
        feature_importance = sorted(zip(X_train.columns, result.importances_mean), key=lambda x: x[1], reverse=True)
        print("Top 5 Feature Importances (Permutation Importance):")
        for feat, imp in feature_importance[:5]:
            print(f"{feat}: {imp:.4f}")
