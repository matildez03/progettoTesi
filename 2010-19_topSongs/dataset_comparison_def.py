#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:40:14 2025

@author: matildezoccolillo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# Caricamento del dataset pulito
df1 = pd.read_csv('../spotify_dataset_2022.csv')

df2 = pd.read_csv("top10s.csv", encoding='ISO-8859-1')

print("DF1:-----------")
print(df1.head())
print(df1.info())

# Rimuovere eventuali valori NaN
df1 = df1.dropna(subset=['popularity'])

# Creare una distribuzione più concentrata sulla seconda metà della scala di popolarità
weights = np.exp(df1['popularity'] / 20)  # Assegna pesi esponenziali per favorire valori alti

df1_sampled = df1.sample(n=603, weights=weights, random_state=42)  # Campionamento ponderato

# Salvare il dataset ridotto
df1_sampled.to_csv("spotify_dataset_2022_reduced.csv", index=False)

# Stampare info per debugging
print("Info di df1_sampled dopo il campionamento:")
print(df1_sampled.info())
print(df1_sampled.head())

df1_sampled["popularity"].hist()
plt.show()

df2["pop"].hist()
plt.show()

# Creare quartili per la classificazione
df1_sampled['popularity_class'] = pd.qcut(df1_sampled['popularity'], q=4, labels=[1, 2, 3, 4])

# Selezionare features e target
X = df1_sampled.drop(columns=['popularity', 'popularity_class','track_id','Unnamed: 0','artists','album_name','track_name','track_genre'])
y = df1_sampled['popularity_class']

# MATRICI DI CORRELAZIONE
# CORRELAZIONE TRA FEATURES NUMERICHE E POPOLARITA'
X1_num = df1_sampled.drop(columns=['popularity','track_id','Unnamed: 0','artists','album_name','track_name','track_genre'])
plt.figure(figsize=(12,8))
sns.heatmap(X1_num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione tra Feature Musicali e Popolarità nel df del 2022")
plt.show()




# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Modelli
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, random_state=42)
}

# Training e valutazione
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance per modelli a base di alberi
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
        print("Top 5 Feature Importances:")
        for feat, imp in feature_importance[:5]:
            print(f"{feat}: {imp:.4f}")

# Salvare il dataset ridotto
df1_sampled.to_csv("spotify_dataset_2022_reduced.csv", index=False)

# Stampare info per debugging
print("Info di df1_sampled dopo il campionamento:")
print(df1_sampled.info())
print(df1_sampled.head())


# Ripeto per df2
print()
print("su df2")
print(df2.info())
print(df2.head())
# Creare quartili per la classificazione
df2['popularity_class'] = pd.qcut(df2['pop'], q=4, labels=[1, 2, 3, 4])

# Selezionare features e target
X = df2.drop(columns=['pop', 'popularity_class','Unnamed: 0','title','artist','top genre'])
y = df2['popularity_class']

# CORRELAZIONE TRA FEATURES NUMERICHE E POPOLARITA'
X2_num = df2.drop(columns=['pop','Unnamed: 0','title','artist','top genre'])
plt.figure(figsize=(12,8))
sns.heatmap(X2_num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione tra Feature Musicali e Popolarità nel df del 2019")
plt.show()

# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Modelli
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, random_state=42)
}

# Training e valutazione
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance per modelli a base di alberi
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
        print("Top 5 Feature Importances:")
        for feat, imp in feature_importance[:5]:
            print(f"{feat}: {imp:.4f}")


# Stampare info per debugging
print("Info di df1_sampled dopo il campionamento:")
print(df1_sampled.info())
print(df1_sampled.head())

