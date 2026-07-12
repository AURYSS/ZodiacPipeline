import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_kmeans(data: pd.DataFrame, features: list, n_clusters: int, model_name: str):
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    
    ensure_dir(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")
    
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    
    return clusters.tolist()

def train_agglomerative(data: pd.DataFrame, features: list, n_clusters: int, linkage: str, model_name: str):
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = agg.fit_predict(X_scaled)
    
    ensure_dir(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")
    
    joblib.dump(agg, model_path)
    joblib.dump(scaler, scaler_path)
    
    return clusters.tolist()

def get_basic_stats(data: pd.DataFrame, features: list):
    stats = {}
    for feature in features:
        if feature in data.columns:
            desc = data[feature].describe()
            stats[feature] = {
                "mean": round(desc.get("mean", 0), 2),
                "std": round(desc.get("std", 0), 2),
                "min": round(desc.get("min", 0), 2),
                "max": round(desc.get("max", 0), 2)
            }
    return stats
