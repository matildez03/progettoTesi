#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:30:29 2025

@author: matildezoccolillo
"""
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import preprocess_data
from eda import (
    plot_popularity_distribution, plot_duration_vs_popularity, plot_loudness_vs_popularity,
    plot_danceability_vs_popularity, plot_speechiness_vs_popularity, plot_tempo_vs_popularity,
    plot_instrumentalness_vs_popularity, plot_correlation_matrix, hist_dataframe,
    plot_danceability_violin, plot_energy_vs_valence, plot_feature_heatmap, plot_radar_chart
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Caricamento del dataset pulito
df = pd.read_csv('spotify_dataset_cleaned.csv')

# Divisione 80% training, 20% test
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['popularity_class'])

features = ['duration_ms','explicit','mode','speechiness','instrumentalness','liveness','tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
target = 'popularity_class'
X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# Matrice di correlazione tra le feature
plot_correlation_matrix(df_train)

#rimozione features inutili
df_train.drop(columns=["mode","key","time_signature","explicit"],inplace=True)

# Creare la nuova feature combinata
df_train["energy_loudness"] = df_train["energy"] * df_train["loudness"]
#df_train["energy_valence"] = df_train["energy"] * df_train["valence"]
df_train["loudness_valence"] = df_train["loudness"] * df_train["valence"]
df_train["danceability_energy"] = df_train["danceability"] * df_train["energy"]
df_train["acousticness_speechiness"] = df_train["acousticness"] * df_train["speechiness"]
df_train["instr_speech_ratio"] = df_train["instrumentalness"] / (df_train["speechiness"] + 1e-5)
df_train["duration_per_bpm"] = df_train["duration_ms"] / df_train["tempo"]
df_train["vocal_intensity"] = df_train["loudness"] * df_train["speechiness"]
df_train["energy_danceability_ratio"] = df_train["energy"] / (df_train["danceability"] + 1e-5)


print(df_train[["instr_speech_ratio", "popularity_class"]].corr())
sns.boxplot(x="popularity_class", y="instr_speech_ratio", data=df_train)
plt.title("Rapporto Instrumentalness/Speechiness vs Popolarità")
plt.show()

print(df_train[["duration_per_bpm", "popularity_class"]].corr())
sns.boxplot(x="popularity_class", y="duration_per_bpm", data=df_train)
plt.title("Rapporto Durata/BPM vs Popolarità")
plt.show()

print(df_train[["vocal_intensity", "popularity_class"]].corr())
sns.boxplot(x="popularity_class", y="vocal_intensity", data=df_train)
plt.title("Vocal Intensity vs Popolarità")
plt.show()

print(df_train[["danceability_energy", "popularity_class"]].corr())
sns.boxplot(x="popularity_class", y="danceability_energy", data=df_train)
plt.title("Danceability*Energy vs Popolarità")
plt.show()

print(df_train[["energy_danceability_ratio", "popularity_class"]].corr())
sns.boxplot(x="popularity_class", y="energy_danceability_ratio", data=df_train)
plt.title("Energy/danceability vs Popolarità")
plt.show()
plot_correlation_matrix(df_train)

# 5 - TRAINING AND CROSS VALIDATION
print("\nInizio training...")

# Cross Validation function
def evaluate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(scores), np.std(scores)

# Addestriamo un modello preliminare per valutare l'importanza delle feature
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)

# Selezioniamo le feature più importanti
feature_importance = pd.Series(rf_temp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
selected_features = feature_importance[:8].index.tolist()  # Prendiamo le prime 8 feature più importanti
X_selected = X_train[selected_features]

print(f"Feature selezionate dopo feature importance: {selected_features}")

# Random Forest con GridSearchCV
param_grid_rf = {
'n_estimators': [100, 300, 500],  # Aumentiamo il numero di alberi
'max_depth': [10, 20, 30],  # Proviamo alberi più profondi
'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Gradient Boosting con GridSearchCV
param_grid_gb = {
'n_estimators': [100, 300],
'learning_rate': [0.01, 0.1, 0.2],  # Aggiungiamo un learning rate più aggressivo
'max_depth': [3, 5, 7]  # Testiamo alberi più profondi
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3, n_jobs=-1)
gb_grid.fit(X_train, y_train)
gb_model = gb_grid.best_estimator_
y_pred_gb = gb_model.predict(X_test)
print("\nGradient Boosting:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(classification_report(y_test, y_pred_gb))

#    Support Vector Machine con GridSearchCV
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}



# Salvare il dataset aggiornato (opzionale)
df_train.to_csv("spotify_dataset_features.csv", index=False)

# Mostrare le prime righe per confermare la nuova colonna
print(df_train.head())