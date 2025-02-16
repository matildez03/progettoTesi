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

    # CLEANING 
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()  # Rimuove spazi dai nomi delle colonne
    
    # Rimozione della colonna non necessaria
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        
    # Rimozione di duplicati
    df_cleaned = df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")


    # CREAZIONE DELLA VARIABILE TARGET
    
    # Creo 5 categorie bilanciate usando i quantili
    df['popularity_class'] = pd.qcut(df['popularity'], q=5, labels=[1, 2, 3, 4, 5])

    # Stampa il numero di istanze in ogni classe
    print(df['popularity_class'].value_counts())

    # Stampa il range di popularity per ogni categoria
    range_by_class = df.groupby('popularity_class')['popularity'].agg(['min', 'max'])
    print(range_by_class)
    
    return df_cleaned

    


