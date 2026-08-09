import os
import pandas as pd
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model_history.csv")

def init_history():
    """Inicializa el archivo de historial si no existe."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "Nombre_Sesion", "Timestamp", "Algoritmo", "Elemento", "K_Clusters", 
            "Inercia", "Silueta", "Ruta_Modelo"
        ])
        df.to_csv(HISTORY_FILE, index=False)

def log_model_training(nombre_sesion, algorithm, elemento, k_clusters, inercia, silueta, model_path, timestamp=None):
    """Guarda un registro de un modelo entrenado."""
    init_history()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    nuevo_registro = {
        "Nombre_Sesion": nombre_sesion,
        "Timestamp": timestamp,
        "Algoritmo": algorithm,
        "Elemento": elemento,
        "K_Clusters": k_clusters,
        "Inercia": round(inercia, 4) if inercia else None,
        "Silueta": round(silueta, 4) if silueta else None,
        "Ruta_Modelo": model_path
    }
    
    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([nuevo_registro])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def get_model_history():
    """Obtiene el historial completo de modelos."""
    init_history()
    return pd.read_csv(HISTORY_FILE)
