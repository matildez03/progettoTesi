#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:25:00 2025

@author: matildezoccolillo
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import preprocess_data
from eda import (
    plot_popularity_distribution, plot_duration_vs_popularity, plot_loudness_vs_popularity, 
    plot_danceability_vs_popularity, plot_speechiness_vs_popularity, plot_tempo_vs_popularity, 
    plot_instrumentalness_vs_popularity, plot_correlation_matrix
)


pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Evita che le colonne vadano a capo

if __name__ == "__main__":
    file_path = "spotify_dataset_2022.csv"  
    
    # 1 - CLEANING AND PREPROCESSING
    df = preprocess_data(file_path)

    # Visualizza le prime righe del dataset pulito
    print("\nDal main:")
    print("Prime righe del df pulito:")
    print(df.head())
    
    print("\nInformazioni sul dataset")
    print(df.info())
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
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['popularity_class'])

    # Controllo delle dimensioni
    print(f"Dimensione Training Set: {df_train.shape[0]} entries")
    print(f"Dimensione Test Set: {df_test.shape[0]} entries")
    
    
    # 3 - EDA
    print("\nAnalisi esplorativa dei dati...")
    
    # Distribuzione della popolarità
    plot_popularity_distribution(df_train)

    # Analizziamo le feature musicali rispetto alla popolarità
    plot_duration_vs_popularity(df_train)
    plot_loudness_vs_popularity(df_train)
    plot_danceability_vs_popularity(df_train)
    plot_speechiness_vs_popularity(df_train)
    plot_tempo_vs_popularity(df_train)
    plot_instrumentalness_vs_popularity(df_train)

    # Matrice di correlazione tra le feature
    plot_correlation_matrix(df_train)
    
    
    