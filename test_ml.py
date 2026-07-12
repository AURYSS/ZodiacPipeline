import pandas as pd
from ml_logic import train_kmeans

df = pd.read_csv('data/perfil_zodiacal_ejemplo.csv')
features = ['Edad', 'Sociabilidad', 'Creatividad', 'Estabilidad_Emocional']
try:
    print("Testing KMeans...")
    clusters = train_kmeans(df, features, 3, "test_kmeans")
    print("KMeans Success!", clusters)
except Exception as e:
    print("Error:", e)
