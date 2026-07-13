import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_clustering_model(df_features, algorithm="kmeans", params=None):
    """
    Entrena el modelo de clustering seleccionado (kmeans, dbscan, gmm).
    Retorna el modelo, las etiquetas del clúster y métricas de desempeño.
    """
    if params is None:
        params = {}
        
    X = df_features.values
    metrics = {}
    model = None
    labels = []
    
    if algorithm == "kmeans":
        n_clusters = params.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        metrics["Inertia"] = float(model.inertia_)
        if len(set(labels)) > 1:
            metrics["Silhouette Score"] = float(silhouette_score(X, labels))
        else:
            metrics["Silhouette Score"] = 0.0
            
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
    X_pca = pca.fit_transform(df_features)
    
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols, index=df_features.index)
    explained_variance = pca.explained_variance_ratio_.tolist()
    
    return df_pca, explained_variance

def save_model(model, algorithm_name):
    """
    Guarda el modelo entrenado en formato .pkl.
    """
    file_path = os.path.join(MODELS_DIR, f"modelo_{algorithm_name}.pkl")
    joblib.dump(model, file_path)
    return file_path

def load_model(algorithm_name):
    """
    Carga el modelo guardado.
    """
    file_path = os.path.join(MODELS_DIR, f"modelo_{algorithm_name}.pkl")
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None
