#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:00:24 2025

@author: matildezoccolillo

feature engineering file definitivo
"""

import pandas as pd
import matplotlib.pyplot as plt
from eda import plot_correlation_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def feature_engineering(sample_size=0.3, top_n_features=13):

    df_train = pd.read_csv("spotify_dataset_train.csv")
    df_test = pd.read_csv("spotify_dataset_test.csv")

    # Matrice di correlazione tra le features e la popolarità prima della feature engineering
    plot_correlation_matrix(df_train)

    # CREAZIONE DI NUOVE FEATURES

    for df in [df_train, df_test]:
           df["energy_loudness"] = df["energy"] * df["loudness"]
           df["loudness_valence"] = df["loudness"] * df["valence"]
           df["danceability_energy"] = df["danceability"] * df["energy"]
           df["acousticness_speechiness"] = df["acousticness"] * df["speechiness"]
           df["speech_instr_ratio"] = df["speechiness"] / (df["instrumentalness"] + 1e-5)
           df["duration_per_bpm"] = df["duration_ms"] / (df["tempo"] + 1e-5)
           df["vocal_intensity"] = df["loudness"] * df["speechiness"]   

    plot_correlation_matrix(df_train)
    
    print(df_train.info())

    # Definizione di feature e target
    # Vengono scartate le features con correlazione più bassa: "key","mode","liveness","tempo","speech_instr_ratio","duration_per_bpm"
    features = ['duration_ms','explicit','speechiness','instrumentalness','energy', 
            'danceability', 'valence', 'acousticness', 
            "energy_loudness","loudness_valence","danceability_energy","acousticness_speechiness","vocal_intensity"]

    target = 'popularity_class'

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]
    
    # Campionamento del dataset per Random Forest
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=(1 - sample_size), random_state=42, stratify=y_train)

    print(f"Campionamento per Feature Importance: {X_train_sample.shape[0]} campioni su {X_train.shape[0]} totali.")

    # Feature Importance con Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sample, y_train_sample)
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    feature_importance.sort_values(ascending=False).plot(kind="bar", figsize=(12, 6))
    plt.title("Importanza delle Feature con Random Forest")
    plt.show()

    # Numero totale di feature disponibili
    num_features_available = len(feature_importance)

    # Se il numero di feature richieste è maggiore di quelle disponibili, lo riduciamo
    if top_n_features > num_features_available:
        print(f"⚠️ Warning: top_n_features={top_n_features} ma ci sono solo {num_features_available} feature disponibili. "
              f"Uso {num_features_available} feature invece.")
        top_n_features = num_features_available
    
    # Selezione delle Top N Feature più importanti
    selected_features = feature_importance.nlargest(top_n_features).index.tolist()
    print(f"{top_n_features} features selezionate: {selected_features}")
    
    # FEATURE SELECTION
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # FEATURES SCALING
    # Creazione dello scaler, fittato solo sui dati di training
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Salvataggio del dataset finale
    pd.DataFrame(X_train_selected, columns=selected_features).to_csv("X_train_selected.csv", index=False)
    pd.DataFrame(X_test_selected, columns=selected_features).to_csv("X_test_selected.csv", index=False)


    # Ritorno dei dataset trasformati
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

# Per eseguire lo script direttamente
if __name__ == "__main__":
    X_train_scaled, y_train, X_test_scaled, y_test, scaler = feature_engineering(sample_size=0.1, top_n_features=10)
