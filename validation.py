#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:50:13 2025

@author: matildezoccolillo
"""
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
from sklearn.model_selection import cross_val_score


# Caricamento del dataset originale
df_train = pd.read_csv('spotify_dataset_features.csv') #questo dataset è ottenuto dalla  divisione in trainig di quello originale

print(df_train.head())

# Supponiamo di avere i tuoi dati di training
X_train = df_train.drop(columns=["popularity_class"])  # Le feature dei tuoi brani
y_train = df_train["popularity_class"]  # La popolarità dei brani suddivisa in 5 livelli

# Inizializziamo il modello
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)


# Applichiamo la cross-validation con 5 folds
scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring="accuracy")

# Stampiamo i risultati
print("Accuratezza per fold:", scores)
print("Media dell'accuratezza:", np.mean(scores))
print("Deviazione standard:", np.std(scores))


# Ora addestriamo il modello su tutto il dataset per ottenere le feature importance
rf_clf.fit(X_train, y_train)

# Estraiamo l'importanza delle feature
feature_importances = rf_clf.feature_importances_

# Creiamo un DataFrame per visualizzarle in ordine decrescente
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Stampiamo le feature più importanti
print("\nFeature Importance:")
print(importance_df)

# Se vuoi visualizzarle in modo più leggibile
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.title("Importanza delle Feature in Random Forest")
plt.gca().invert_yaxis()  # Inverti l'asse y per avere le più importanti in alto
plt.show()