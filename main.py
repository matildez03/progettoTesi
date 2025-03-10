#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:25:00 2025

@author: matildezoccolillo

file da eseguire
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from eda import (
    plot_popularity_distribution, plot_duration_vs_popularity, plot_loudness_vs_popularity,
    plot_danceability_vs_popularity, plot_speechiness_vs_popularity, plot_tempo_vs_popularity,
    plot_instrumentalness_vs_popularity, plot_correlation_matrix, hist_dataframe,
    plot_feature_heatmap, plot_radar_chart
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from feature_engineering import feature_engineering
from hyperparameter_tuning import optimize_hyperparameters
from lightgbm import LGBMClassifier
from model_test import test_model
from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False) # Evita che le colonne vadano a capo

if __name__ == "__main__":
    file_path = "spotify_dataset_2022.csv"

    # 1 - CLEANING AND PREPROCESSING
    df = preprocess_data(file_path)
    
    # 1.1 - QUICK LOOK AT THE DATA STRUCTURE

    print("\nFile: main.py")
    print("\n")
    print("Prime righe del df pulito:")
    print(df.head())

    print("\nInformazioni sul dataset pulito")
    print("\n", df.info())
    
    print("\nStatistiche descrittive sul dataset pulito:")
    print("\n", df.describe())

    # Controllo e visualizzazione della distribuzione delle classi target
        # Stampa il numero di istanze in ogni classe
    print(df['popularity_class'].value_counts())

    
        # Istogramma della distribuzione
    plot_popularity_distribution(df)

    # 2 - DIVISIONE IN TRAINING E TEST SET
    # avvenuta in data_preprocessing
    df_train = pd.read_csv("spotify_dataset_train.csv")
    df_test = pd.read_csv("spotify_dataset_test.csv")

    # Controllo delle dimensioni
    print(f"Dimensione Training Set: {df_train.shape[0]} entries")
    print(f"Dimensione Test Set: {df_test.shape[0]} entries")
    print("")
    

    # 3 - EDA
    print("\nAnalisi esplorativa dei dati...")
    print(df_train.info())

    # Istogrammi delle features musicali
    hist_dataframe(df_train)

    # Distribuzione della popolarità
    plot_popularity_distribution(df_train)

    # Analizzo le feature musicali rispetto alla popolarità
    plot_duration_vs_popularity(df_train)
    plot_loudness_vs_popularity(df_train)
    plot_danceability_vs_popularity(df_train)
    plot_speechiness_vs_popularity(df_train)
    plot_tempo_vs_popularity(df_train)
    plot_instrumentalness_vs_popularity(df_train)
    

    # Matrice di correlazione tra le features e la popolarità
    plot_correlation_matrix(df_train)

    # Heatmap delle features musicali più presenti in ogni classe di popolarità
    plot_feature_heatmap(df_train)

    # Grafico radiale delle features per ogni classe di popolarità
    plot_radar_chart(df_train)
    
    
    # 4 - FEATURE ENGINEERING
    print("Feature engineering...")
    X_train, y_train, X_test, y_test, scaler, features = feature_engineering(0.3, 21)


    # 5 - HYPERPARAMETERS TUNING
    print("Inizio dell'ottimizzazione degli iperparametri e del training...")

    
    # RANODM FOREST
    # Definizione dei parametri per Random Forest
    param_grid_rf = {
        'n_estimators': [500, 1000],
        'max_depth': [30, 50,100],
        'min_samples_split': [2, 5, 10]
    }

    # Ottimizzazione Random Forest
    rf_model = optimize_hyperparameters(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train)
    
    # LIGHTGBM
    param_grid_lgb = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.001, 0.01],
    'max_depth': [30, 50, 100],
    'num_leaves': [31, 128]
    }
    lgb_model = optimize_hyperparameters(LGBMClassifier(random_state=42), param_grid_lgb, X_train, y_train)

    
    # SUPPORT VECTOR MACHINE
    # Definizione dei parametri per SVM
    param_grid_svm = {
        'C': [1, 10, 20],  
        'kernel': ['rbf', 'linear']  
    }


    # Ottimizzazione SVM
    svm_model = optimize_hyperparameters(SVC(random_state=42), param_grid_svm, X_train, y_train)
    
    
    # 6 - TESTING
    print("\n########## RISULTATI TEST ##########")
     
    # Test Light GBM
    test_model(lgb_model, "LightGBM", X_test, y_test)

    # Test Random Forest
    test_model(rf_model, "Random Forest", X_test, y_test)
    
    print("Importanza delle features nelle previsioni RF:")
    # Feature Importance con Random Forest
    feature_importance = pd.Series(rf_model.feature_importances_, index=features)
    feature_importance.sort_values(ascending=False).plot(kind="bar", figsize=(12, 6))
    plt.title("Importanza delle Feature con Random Forest")
    plt.show()
    
    # Test SVM
    test_model(svm_model, "Support Vector Machine (SVM)", X_test, y_test)
    
    
    


    print("\n########## TEST COMPLETATO ##########")

    
    
