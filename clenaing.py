#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:47:36 2025

@author: matildezoccolillo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder

#formattazione della stampa di pandas
pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
pd.set_option('display.expand_frame_repr', False)  # Evita che le colonne vadano a capo

# Carico il dataset
file_path = "spotify_dataset_2022.csv"  
df = pd.read_csv(file_path)

df.dropna(inplace=True)
df.columns = df.columns.str.strip()  # Rimuove spazi dai nomi delle colonne
df.drop(columns=["Unnamed: 0"], inplace=True)

print(df.info())

#controllo se ci sono duplicati di intere righe
duplicate_rows = df[df.duplicated()]
print(f"Numero di righe duplicate: {duplicate_rows.shape[0]}")

#controllo la presenza di duplicati relativi solo all'id del brano
duplicate_rows = df[df["track_id"].duplicated()]
print(f"Numero di id duplicati: {duplicate_rows.shape[0]}")

# Identifica solo gli ID duplicati
duplicated_ids = df[df.duplicated(keep=False)]

# Verifica quali colonne contengono valori diversi per lo stesso ID
diff_columns = duplicated_ids.groupby('track_id').nunique().gt(1)
columns_with_diff = diff_columns.columns[diff_columns.any()].tolist()

# Mostra solo le colonne che contengono valori diversi
df_diff = duplicated_ids[['track_id'] + columns_with_diff].sort_values(by='track_id')

# Mostra le differenze
print(df_diff)

# Analizzo le differenze per popolarità
pop_diff = duplicated_ids.groupby("track_id")["popularity"].nunique().sort_values()
pop_diff = pop_diff[pop_diff > 1]  # Filtra solo quelli con più di un valore unico

print(pop_diff)

#ci sono 87 entries relative a brani uguali con popolarità diverse, faccio un controllo più specifico:
    
print(df[df["track_id"] == "4KROoGIaPaR1pBHPnR3bwC"][["track_id", "popularity"]])

print(df[df["track_id"] == "4ZlHgEGwZb3PCq0OWFbhcO"][["track_id", "popularity"]])

print(df[df["track_id"] == "4uJnQswt7f4r0tuI37wNFk"][["track_id", "popularity"]])

print(df[df["track_id"] == "1sZyestTGfOxWKRxdVdKuA"][["track_id", "popularity"]])

#df_after = df[~df.duplicated(subset="track_id", keep=False)]
df_after =df.sort_values(by="popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")

print(df_after.info())

df["popularity"].hist(alpha=0.5, label="Prima", bins=20)
df_after["popularity"].hist(alpha=0.5, label="Dopo", bins=20)
plt.legend()
plt.show()

#controllo se ci sono duplicati di intere righe
duplicate_rows = df_after[df_after.duplicated()]
print(f"Numero di righe duplicate: {duplicate_rows.shape[0]}")

#controllo la presenza di duplicati relativi solo all'id del brano
duplicate_rows = df_after[df_after["track_id"].duplicated()]
print(f"Numero di id duplicati: {duplicate_rows.shape[0]}")

print(df_after.info())
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Differenze tra ID duplicati", dataframe=df_diff)

# # Seleziona solo gli ID che appaiono più di una volta
# duplicated_ids = df[df.duplicated(subset=['track_id'], keep=False)]

# # Ordina per ID per una migliore visualizzazione
# duplicated_ids = duplicated_ids.sort_values(by='track_id')

# # Mostra solo gli ID duplicati e le loro differenze
# print("Istanze duplicate:")
# print(duplicated_ids)

# # Identifica quali colonne hanno valori diversi per lo stesso ID
# diff_columns = duplicated_ids.groupby('track_id').nunique().gt(1)
# columns_with_diff = diff_columns.columns[diff_columns.any()].tolist()

# # Mostra solo le colonne che hanno valori diversi
# df_diff = duplicated_ids[['track_id'] + columns_with_diff].sort_values(by='track_id')
# print(df_diff)

# # Crea un dizionario che mostra i valori unici per ogni ID duplicato
# diff_summary = duplicated_ids.groupby('track_id').agg(lambda x: x.unique().tolist())

# # Mostra solo le colonne con differenze
# diff_summary = diff_summary.loc[:, diff_summary.nunique() > 1]
# print(diff_summary)