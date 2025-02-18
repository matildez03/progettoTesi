#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:25:00 2025

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
# Evita che le colonne vadano a capo
pd.set_option('display.expand_frame_repr', False)

if __name__ == "__main__":
    file_path = "spotify_dataset_2022.csv"

    # 1 - CLEANING AND PREPROCESSING
    df = preprocess_data(file_path)

    # Visualizza le prime righe del dataset pulito
    print("\nDal main:")
    print("Prime righe del df pulito:")
    print(df.head())

    print("\nInformazioni sul dataset")
    print("\n", df.info())

    # Controlliamo la distribuzione
    print("Numero di valori per ciascuna classe target:")
    print(df['popularity_class'].value_counts())

    # Visualizziamo la distribuzione dopo la suddivisione
    df['popularity_class'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Classi di Popolarità')
    plt.ylabel('Numero di brani')
    plt.title('Distribuzione delle Classi di Popolarità')
    plt.show()

    # 2 - DIVISIONE IN TRAINING E TEST SET
    # Divisione 80% training, 20% test
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['popularity_class'])

    # Controllo delle dimensioni
    print(f"Dimensione Training Set: {df_train.shape[0]} entries")
    print(f"Dimensione Test Set: {df_test.shape[0]} entries")

    # 3 - EDA
    print("\nAnalisi esplorativa dei dati...")

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
    plot_danceability_violin(df_train)
    plot_energy_vs_valence(df_train)

    # Matrice di correlazione tra le feature
    plot_correlation_matrix(df_train)

    # Heatmap delle features musicali più presenti in ogni classe di popolarità
    plot_feature_heatmap(df_train)

    # Grafico radiale delle features per ogni classe di popolarità
    plot_radar_chart(df_train)

    # 4 - SCALING
    # Separo feature e target
    features = ['duration_ms','explicit','mode','speechiness','instrumentalness','liveness','tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
    target = 'popularity_class'
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    # Creo lo scaler, fittato solo sui dati di training
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Trasformo il test set con lo stesso scaler (senza rifare il fit)
    X_test_scaled = scaler.transform(X_test)

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
    # svm_grid = GridSearchCV(SVC(), param_grid_svm, cv=3, n_jobs=-1)
    # svm_grid.fit(X_train, y_train)
    # svm_model = svm_grid.best_estimator_
    # y_pred_svm = svm_model.predict(X_test)
    # print("\nSVM:")
    # print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    # print(classification_report(y_test, y_pred_svm))
