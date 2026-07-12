"""
modelos.py
----------
Entrenamiento, persistencia y uso de los dos algoritmos no supervisados
elegidos para el aplicativo:

    1. K-Means            (sklearn.cluster.KMeans)
    2. Clusterización Jerárquica  (sklearn.cluster.AgglomerativeClustering)

Ambos modelos (junto con el StandardScaler usado para normalizar los
datos) se guardan en disco con joblib, para poder reutilizarlos después
sin necesidad de reentrenar ("guardar para uso posterior").
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica, necesario en servidor
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(MODELS_DIR, exist_ok=True)


def _extraer_matriz(registros, columnas):
    """Convierte una lista de dicts (filas) en una matriz numérica X."""
    X = []
    for fila in registros:
        try:
            X.append([float(fila[col]) for col in columnas])
        except (KeyError, TypeError, ValueError):
            continue
    return np.array(X)


def entrenar_kmeans(registros, columnas, k=4):
    """Entrena K-Means sobre las columnas numéricas indicadas."""
    X = _extraer_matriz(registros, columnas)
    if len(X) < k:
        raise ValueError("No hay suficientes registros para el número de clústeres solicitado.")

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    etiquetas = modelo.fit_predict(X_esc)

    silueta = silhouette_score(X_esc, etiquetas) if k > 1 and len(set(etiquetas)) > 1 else None

    resultado = {
        "algoritmo": "kmeans",
        "k": k,
        "columnas": columnas,
        "etiquetas": etiquetas.tolist(),
        "centroides": scaler.inverse_transform(modelo.cluster_centers_).round(2).tolist(),
        "inercia": round(float(modelo.inertia_), 2),
        "silueta": round(float(silueta), 3) if silueta is not None else None,
    }
    return modelo, scaler, resultado


def entrenar_jerarquico(registros, columnas, k=4, metodo_enlace="ward"):
    """Entrena Clusterización Jerárquica Aglomerativa y genera el dendrograma."""
    X = _extraer_matriz(registros, columnas)
    if len(X) < k:
        raise ValueError("No hay suficientes registros para el número de clústeres solicitado.")

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo_enlace)
    etiquetas = modelo.fit_predict(X_esc)

    silueta = silhouette_score(X_esc, etiquetas) if k > 1 and len(set(etiquetas)) > 1 else None

    # Generar y guardar el dendrograma como imagen para mostrarlo en el frontend
    matriz_enlace = linkage(X_esc, method=metodo_enlace)
    plt.figure(figsize=(10, 5))
    dendrogram(matriz_enlace, truncate_mode="lastp", p=20, leaf_rotation=90)
    plt.title("Dendrograma - Clusterización Jerárquica")
    plt.xlabel("Registros agrupados")
    plt.ylabel("Distancia")
    plt.tight_layout()
    ruta_imagen = os.path.join(STATIC_DIR, "dendrograma.png")
    plt.savefig(ruta_imagen, dpi=110)
    plt.close()

    resultado = {
        "algoritmo": "jerarquico",
        "k": k,
        "metodo_enlace": metodo_enlace,
        "columnas": columnas,
        "etiquetas": etiquetas.tolist(),
        "silueta": round(float(silueta), 3) if silueta is not None else None,
        "dendrograma_url": "/static/dendrograma.png",
    }
    return modelo, scaler, resultado


def guardar_modelo(modelo, scaler, columnas, nombre_archivo):
    """Guarda el modelo entrenado y su scaler para uso posterior."""
    ruta = os.path.join(MODELS_DIR, f"{nombre_archivo}.pkl")
    joblib.dump({"modelo": modelo, "scaler": scaler, "columnas": columnas}, ruta)
    return ruta


def cargar_modelo(nombre_archivo):
    """Carga un modelo previamente guardado."""
    ruta = os.path.join(MODELS_DIR, f"{nombre_archivo}.pkl")
    if not os.path.exists(ruta):
        return None
    return joblib.load(ruta)


def listar_modelos_guardados():
    """Lista los modelos disponibles en disco."""
    if not os.path.exists(MODELS_DIR):
        return []
    return [f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]


def clasificar_nuevo_registro(nombre_archivo, valores_dict):
    """
    Usa un modelo ya entrenado y guardado para asignar un clúster a un
    registro nuevo (por ejemplo, un usuario que acaba de llenar su perfil).
    Solo aplica a K-Means, ya que la Clusterización Jerárquica no tiene
    un método .predict() nativo para nuevas observaciones.
    """
    paquete = cargar_modelo(nombre_archivo)
    if paquete is None:
        raise FileNotFoundError("El modelo indicado no existe. Entrena y guarda un modelo primero.")

    modelo = paquete["modelo"]
    scaler = paquete["scaler"]
    columnas = paquete["columnas"]

    if not hasattr(modelo, "predict"):
        raise TypeError("Este modelo no soporta clasificación de nuevos registros (usa K-Means para esto).")

    X_nuevo = np.array([[float(valores_dict[col]) for col in columnas]])
    X_esc = scaler.transform(X_nuevo)
    cluster = int(modelo.predict(X_esc)[0])
    return cluster
