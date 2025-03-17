#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: matildezoccolillo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Evita che le colonne vadano a capo


def preprocess_data(file_path):
    """
    Carica e pre-elabora il dataset, rimuovendo dati mancanti e duplicati,
    creando la variabile target della popolarità suddivisa in 4 classi.
    
    Args:
        file_path (str): Percorso del file CSV da caricare.
    
    Returns:
        pd.DataFrame: DataFrame pulito e con la variabile target aggiunta.
    """
    
    # Caricamento del dataset
    df = pd.read_csv(file_path)
    
    print("Informazioni sul dataset:")
    print(df.info())
    
    #distribuzione di popolarità
    df["popularity"].hist()
    plt.show()

    # CLEANING 
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    
    # Rimozione di duplicati
    df_cleaned = df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")
    
    # Rimozione colonne non utili per la classificazione basata su dati musicali
    df_cleaned.drop(columns=['Unnamed: 0','track_name', 'artists', 'album_name', 'track_genre'], inplace=True)

    # Converte explicit in numerico 
    df_cleaned['explicit'] = df_cleaned['explicit'].astype(int)


    # CREAZIONE DELLA VARIABILE TARGET
    # Creo 4 categorie bilanciate usando i quartili
    df_cleaned['popularity_class'] = pd.qcut(df_cleaned['popularity'], q=4, labels=[1, 2, 3, 4])
    
    # Stampa il range di popularity per ogni categoria
    range_by_class = df_cleaned.groupby('popularity_class')['popularity'].agg(['min', 'max'])
    print(range_by_class)

    # Salva i file csv del dataset pulito con e senza la popolarità numerica
    df_cleaned.to_csv("spotify_dataset_cleaned_with_popularity.csv", index=False)
    df_cleaned.drop(columns=["popularity","track_id"], inplace=True)
    df_cleaned.to_csv("spotify_dataset_cleaned.csv", index=False)

    # Divisione 80% training, 20% test
    df_train, df_test = train_test_split(df_cleaned, test_size=0.2, random_state=42, stratify=df_cleaned['popularity_class'])
    
    df_train.to_csv("spotify_dataset_train.csv", index=False)
    df_test.to_csv("spotify_dataset_test.csv", index=False)
    
    print("train and test csv files created")
    print("End of data preprocessing")
    return df_cleaned

    


