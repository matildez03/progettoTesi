#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:11:56 2025

@author: matildezoccolillo
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def hist_dataframe(df):
    """Istogramma dei valori numerici del dataframe"""
    plt.figure(figsize=(8,5))
    df.hist()
    plt.title('Istogrammi delle variabili numeriche')
    plt.show()

def plot_popularity_distribution(df):
    """Distribuzione delle classi di popolarità."""
    plt.figure(figsize=(8,5))
    sns.countplot(x=df['popularity_class'], palette="viridis")
    plt.xlabel('Classi di Popolarità')
    plt.ylabel('Numero di Brani')
    plt.title('Distribuzione delle Classi di Popolarità')
    plt.show()

def plot_duration_vs_popularity(df):
    """Boxplot della durata del brano rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['duration_ms'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Durata (ms)")
    plt.title("Durata del Brano vs Popolarità")
    plt.show()

def plot_loudness_vs_popularity(df):
    """Boxplot del volume (loudness) rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['loudness'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Loudness (dB)")
    plt.title("Loudness vs Popolarità")
    plt.show()

def plot_danceability_vs_popularity(df):
    """Boxplot della danceability rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['danceability'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Danceability")
    plt.title("Danceability vs Popolarità")
    plt.show()

def plot_speechiness_vs_popularity(df):
    """Boxplot della speechiness rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['speechiness'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Speechiness")
    plt.title("Speechiness vs Popolarità")
    plt.show()

def plot_tempo_vs_popularity(df):
    """Boxplot del tempo rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['tempo'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Tempo (BPM)")
    plt.title("Tempo (BPM) vs Popolarità")
    plt.show()

def plot_instrumentalness_vs_popularity(df):
    """Boxplot della instrumentalness rispetto alla popolarità."""
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df['popularity_class'], y=df['instrumentalness'])
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Instrumentalness")
    plt.title("Strumentalità vs Popolarità")
    plt.show()

def plot_correlation_matrix(df):
    """Matrice di correlazione tra le feature musicali e la popolarità."""
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice di Correlazione tra Feature Musicali e Popolarità")
    plt.show()
    
def plot_feature_heatmap(df):
    """Crea una heatmap con la media delle feature musicali per classe di popolarità."""
    feature_means = df.groupby('popularity_class').mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(feature_means, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmap delle Feature Musicali per Classe di Popolarità")
    plt.xlabel("Feature Musicali")
    plt.ylabel("Classe di Popolarità")
    plt.show()

def plot_danceability_violin(df):
    """Violin plot per mostrare la distribuzione della danceability per classe di popolarità."""
    plt.figure(figsize=(12,6))
    sns.violinplot(x=df['popularity_class'], y=df['danceability'], palette="mako")
    plt.xlabel("Classe di Popolarità")
    plt.ylabel("Danceability")
    plt.title("Distribuzione della Danceability per Classe di Popolarità")
    plt.show()
    

def plot_radar_chart(df):
    """Grafico radiale per confrontare le feature musicali tra le classi di popolarità."""
    feature_means = df.groupby("popularity_class").mean()
    categories = feature_means.columns

    # Normalizziamo i valori tra 0 e 1 per una migliore visualizzazione
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min())

    plt.figure(figsize=(8, 8))
    
    for pop_class in feature_means.index:
        values = feature_means.loc[pop_class].values.flatten().tolist()
        values += values[:1]  # Chiudiamo il cerchio

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, label=f"Classe {pop_class}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    plt.title("Grafico Radiale delle Feature Musicali per Classe di Popolarità")
    plt.legend()
    plt.show()
    
def plot_energy_vs_valence(df):
    """Scatter plot di Energy vs Valence colorato per classe di popolarità."""
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, x="energy", y="valence", hue="popularity_class", palette="coolwarm", alpha=0.6)
    plt.xlabel("Energia")
    plt.ylabel("Valenza Emotiva")
    plt.title("Energy vs Valence nei Brani Popolari")
    plt.legend(title="Classe di Popolarità")
    plt.show()


