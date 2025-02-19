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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import preprocess_data
from eda import (
    plot_popularity_distribution, plot_duration_vs_popularity, plot_loudness_vs_popularity,
    plot_danceability_vs_popularity, plot_speechiness_vs_popularity, plot_tempo_vs_popularity,
    plot_instrumentalness_vs_popularity, plot_correlation_matrix, hist_dataframe,
    plot_danceability_violin, plot_energy_vs_valence, plot_feature_heatmap, plot_radar_chart
)

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
df_train["energy_valence"] = df_train["energy"] * df_train["valence"]
df_train["loudness_valence"] = df_train["loudness"] * df_train["valence"]
df_train["danceability_energy"] = df_train["danceability"] * df_train["energy"]
df_train["acousticness_speechiness"] = df_train["acousticness"] * df_train["speechiness"]

plot_correlation_matrix(df_train)



# Salvare il dataset aggiornato (opzionale)
df_train.to_csv("spotify_dataset_features.csv", index=False)

# Mostrare le prime righe per confermare la nuova colonna
print(df_train.head())