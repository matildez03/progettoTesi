#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:15:42 2025

@author: matildezoccolillo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Caricamento del dataset pulito
df1 = pd.read_csv('../spotify_dataset_2022.csv')

df2 = pd.read_csv("top10s.csv", encoding='ISO-8859-1')

print("DF1:-----------")
print(df1.head())
print(df1.info())


print("DF2:-----------")
print(df2.head())
print(df2.info())


# Rimuovere eventuali valori NaN
df1 = df1.dropna(subset=['popularity'])
df2 = df2.dropna(subset=['pop'])

# Creare bin di popolarità per entrambi i dataset
num_bins = min(15, df1['popularity'].nunique(), df2['pop'].nunique())  # Assicurarsi che ci siano abbastanza bin
df1['pop_bin'] = pd.qcut(df1['popularity'], q=num_bins, duplicates='drop')
df2['pop_bin'] = pd.qcut(df2['pop'], q=num_bins, duplicates='drop')

# Verificare che la colonna pop_bin esista
df1.dropna(subset=['pop_bin'], inplace=True)
df2.dropna(subset=['pop_bin'], inplace=True)

# Contare la distribuzione nel dataset più piccolo
bin_counts = df2['pop_bin'].value_counts(normalize=True)

# Campionare 603 canzoni da df1 mantenendo la stessa distribuzione
samples_per_bin = (603 * bin_counts).astype(int)

# Assicurarsi che ogni bin abbia almeno 1 campione
def sample_bin(group, bin_name):
    n = samples_per_bin.get(bin_name, 0)
    return group.sample(n=min(len(group), n), random_state=42) if n > 0 else None

df1_sampled = df1.groupby('pop_bin', group_keys=False).apply(lambda x: sample_bin(x, x.name))
df1_sampled = df1_sampled.dropna().sample(n=min(603, len(df1_sampled)), random_state=42)  # Garantire esattamente 603 entries

print(df1_sampled.info())
# Rimuovere la colonna temporanea dei bin
df1_sampled.drop(columns=['pop_bin'], inplace=True, errors='ignore')

# Verificare che 'popularity' sia ancora presente
if 'popularity' not in df1_sampled.columns:
    print("Errore: La colonna 'popularity' non è presente in df1_sampled.")
    print("Colonne disponibili:", df1_sampled.columns)
else:
    # Visualizzare la distribuzione dopo il campionamento
    plt.figure(figsize=(8, 6))
    df1_sampled['popularity'].hist(bins=15, alpha=0.7, color='green', label='Spotify Sampled')
    df2['pop'].hist(bins=15, alpha=0.7, color='blue', label='Top10s')
    plt.title("Distribuzione della Popolarità dopo il campionamento")
    plt.xlabel("Popolarità")
    plt.ylabel("Numero di canzoni")
    plt.legend()
    plt.show()

# Salvare il nuovo dataset ridotto
df1_sampled.to_csv("spotify_dataset_2022_reduced.csv", index=False)
