import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def calcular_distancia_euclidiana(p1, p2):
    """Calcula la distancia euclidiana manual entre dos puntos."""
    suma_cuadrados = 0.0
    for v1, v2 in zip(p1, p2):
        suma_cuadrados += (v1 - v2)**2
    return suma_cuadrados ** 0.5

def calcular_inercia_manual(X, labels, centroids):
    """Calcula la inercia (suma de distancias al cuadrado a los centroides) manualmente."""
    inercia = 0.0
    for i, punto in enumerate(X):
        centroide = centroids[labels[i]]
        # Inercia es distancia al cuadrado
        suma_cuadrados = 0.0
        for v1, v2 in zip(punto, centroide):
            suma_cuadrados += (v1 - v2)**2
        inercia += suma_cuadrados
    return inercia

def calcular_silueta_manual(X, labels):
    """Calcula el coeficiente de silueta de forma manual."""
    n_samples = len(X)
    if n_samples < 2 or len(set(labels)) < 2:
        return 0.0
        
    unique_labels = np.unique(labels)
    s_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        x_i = X[i]
        label_i = labels[i]
        
        # Calcular a(i): distancia media a otros puntos del MISMO clúster
        same_cluster_mask = (labels == label_i)
        same_cluster_mask[i] = False
        n_same = np.sum(same_cluster_mask)
        if n_same == 0:
            a_i = 0.0
        else:
            diffs = X[same_cluster_mask] - x_i
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            a_i = np.mean(dists)
            
        # Calcular b(i): distancia media mínima a puntos de OTRO clúster
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster_mask = (labels == other_label)
            diffs = X[other_cluster_mask] - x_i
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            avg_dist = np.mean(dists)
            if avg_dist < b_i:
                b_i = avg_dist
                
        if a_i == 0.0 and b_i == float('inf'):
            s_vals[i] = 0.0
        else:
            s_vals[i] = (b_i - a_i) / max(a_i, b_i)
            
    return float(np.mean(s_vals))


def train_clustering_model(df_features, algorithm="kmeans", params=None):
    """
    Entrena el modelo de clustering seleccionado (kmeans, dbscan, gmm).
    Retorna el modelo, las etiquetas del clúster y métricas de desempeño.
    """
    if params is None:
        params = {}
        
    # Imputar valores nulos con 3 (Neutral) antes de entrenar en sklearn
    df_filled = df_features.fillna(3)
    X = df_filled.values
    metrics = {}
    model = None
    labels = []
    
    if algorithm == "kmeans":
        n_clusters = params.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_
        
        # Uso de fórmulas manuales en lugar de las de sklearn
        metrics["Inertia (Manual)"] = float(calcular_inercia_manual(X, labels, centroids))
        if len(set(labels)) > 1:
            metrics["Silhouette Score (Manual)"] = calcular_silueta_manual(X, labels)
        else:
            metrics["Silhouette Score (Manual)"] = 0.0
            
    elif algorithm == "dbscan":
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1 and len(labels[labels != -1]) > 1:
            metrics["Silhouette Score"] = float(silhouette_score(X[labels != -1], labels[labels != -1]))
        else:
            metrics["Silhouette Score"] = 0.0
        metrics["Noise points"] = int(np.sum(labels == -1))
        
    elif algorithm == "gmm":
        n_components = params.get("n_components", 3)
        model = GaussianMixture(n_components=n_components, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        metrics["BIC"] = float(model.bic(X))
        metrics["AIC"] = float(model.aic(X))
        if len(set(labels)) > 1:
            metrics["Silhouette Score"] = float(silhouette_score(X, labels))
        else:
            metrics["Silhouette Score"] = 0.0
            
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
        
    return model, labels, metrics

def apply_pca_reduction(df_features, n_components=3):
    """
    Aplica PCA sobre las características para reducción de dimensionalidad (2D o 3D).
    Retorna un DataFrame con los componentes.
    """
    pca = PCA(n_components=n_components, random_state=42)
    # Imputar nulos con 3 (Neutral) antes de entrenar PCA
    df_filled = df_features.fillna(3)
    X_pca = pca.fit_transform(df_filled)
    
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols, index=df_features.index)
    explained_variance = pca.explained_variance_ratio_.tolist()
    
    return df_pca, explained_variance

def save_model(model, algorithm_name, elemento_name=None):
    """
    Guarda el modelo entrenado en formato .pkl con un timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{elemento_name.lower()}_{timestamp}" if elemento_name else f"_{timestamp}"
    file_path = os.path.join(MODELS_DIR, f"modelo_{algorithm_name}{suffix}.pkl")
    joblib.dump(model, file_path)
    return file_path

def load_model(algorithm_name, elemento_name=None):
    """
    Carga el modelo guardado.
    """
    suffix = f"_{elemento_name.lower()}" if elemento_name else ""
    file_path = os.path.join(MODELS_DIR, f"modelo_{algorithm_name}{suffix}.pkl")
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None
