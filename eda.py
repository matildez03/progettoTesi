#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:11:56 2025

@author: matildezoccolillo
"""

import matplotlib.pyplot as plt
import seaborn as sns



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

