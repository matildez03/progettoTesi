#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:21:00 2025

@author: matildezoccolillo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#formattazione della stampa di pandas
pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Evita che le colonne vadano a capo

# Carico il dataset
file_path = "spotify_dataset_2022.csv"  
df = pd.read_csv(file_path)

# Mostra le prime righe del dataset
print("Prime righe del dataset:")
print(df.head())

# Controllo il tipo di dati e i valori mancanti
print("\nInformazioni sul dataset:")
print(df.info())

# CLEANING 
df.dropna(inplace=True)
df.columns = df.columns.str.strip()  # Rimuove spazi dai nomi delle colonne
df.drop(columns=["Unnamed: 0"], inplace=True)

# Controllo la presenza di duplicati relativi solo all'id del brano
duplicate_rows = df[df["track_id"].duplicated()]
print(f"Numero di id duplicati: {duplicate_rows.shape[0]}")

df_cleaned =df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")

print(df_cleaned.info())
# Controllo dopo la pulizia
print(df_cleaned.info())
# Controllo la presenza di duplicati relativi solo all'id del brano
duplicate_rows = df_cleaned[df_cleaned["track_id"].duplicated()]
print(f"Numero di id duplicati: {duplicate_rows.shape[0]}")

df_cleaned.hist()
plt.show()

df_cleaned["popularity"].hist()
plt.show()

# CREAZIONE DELLA VARIABILE TARGET
# Creo 5 categorie bilanciate usando i quantili
df['popularity_class'] = pd.qcut(df['popularity'], q=5, labels=[1, 2, 3, 4, 5])

# Contiamo il numero di istanze in ogni classe
print(df['popularity_class'].value_counts())

range_by_class = df.groupby('popularity_class')['popularity'].agg(['min', 'max'])
print(range_by_class)

# Controlliamo la distribuzione
print(df['popularity_class'].value_counts())
# Visualizziamo la distribuzione dopo la suddivisione
df['popularity_class'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Classi di Popolarità')
plt.ylabel('Numero di brani')
plt.title('Distribuzione delle Classi di Popolarità')
plt.show()


