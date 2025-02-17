#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:33:28 2025

@author: matildezoccolillo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Evita che le colonne vadano a capo


def preprocess_data(file_path):
    """
    Carica e pre-elabora il dataset di Spotify, rimuovendo dati mancanti e duplicati,
    creando la variabile target della popolarità suddivisa in 5 classi.
    
    Args:
        file_path (str): Percorso del file CSV da caricare.
    
    Returns:
        pd.DataFrame: DataFrame pulito e con la variabile target aggiunta.
    """
    
    # Carico il dataset
    df = pd.read_csv(file_path)
    
    print(df.head())
    print(df.info())

    # CLEANING 
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()  # Rimuove spazi dai nomi delle colonne

        
    # Rimozione colonne non utili per la classificazione basata su dati musicali
    df.drop(columns=['Unnamed: 0','track_name', 'artists', 'album_name', 'track_genre'], inplace=True)

        
    # Rimozione di duplicati
    df_cleaned = df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")

    # Converte explicit in numerico 
    df_cleaned['explicit'] = df_cleaned['explicit'].astype(int)

    # CREAZIONE DELLA VARIABILE TARGET
    
    # Creo 5 categorie bilanciate usando i quantili
    df_cleaned['popularity_class'] = pd.qcut(df_cleaned['popularity'], q=4, labels=[1, 2, 3, 4])
    
    # Stampa il numero di istanze in ogni classe
    print(df_cleaned['popularity_class'].value_counts())

    # Stampa il range di popularity per ogni categoria
    range_by_class = df_cleaned.groupby('popularity_class')['popularity'].agg(['min', 'max'])
    print(range_by_class)
    
    
    # Salva i file csv del dataset pulito con e senza la popolarità numerica
    df_cleaned.to_csv("spotify_dataset_cleaned_with_popularity.csv", index=False)
    df_cleaned.drop(columns=["popularity","track_id"], inplace=True)
    df_cleaned.to_csv("spotify_dataset_cleaned.csv", index=False)
    
    return df_cleaned

    


