import os
import sys
import pandas as pd

# Añadir directorio actual al path para importar módulos correctamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.database import obtener_todas_las_encuestas
from modules.statistics import normalizacion_min_max_manual
from modules.clustering import train_clustering_model, save_model

def main():
    print("====== Script Autónomo de Aprendizaje de Modelos (Zodiac) ======")
    
    # 1. Cargar datos de Postgres
    try:
        df = obtener_todas_las_encuestas()
    except Exception as e:
        print(f"❌ Error al conectar a PostgreSQL o recuperar encuestas: {e}")
        print("Asegúrese de que el servidor Postgres esté activo, la BD 'zodiac' exista y contenga registros.")
        sys.exit(1)
        
    if df.empty:
        print("⚠️ La base de datos está vacía. Inserte datos a través de la App web primero o cargue el dataset de prueba.")
        sys.exit(0)
        
    print(f"📊 Registros recuperados de la BD: {len(df)}")
    
    # Definir características para entrenamiento (p1 a p15)
    features = [f"p{i}" for i in range(1, 16)]
    
    # Asegurar que existan las columnas
    features = [f for f in features if f in df.columns]
    
    print(f"⚙️ Características de entrenamiento: {features}")
    df_feat = df[features].copy()
    
    # 2. Normalización Min-Max Manual (Algoritmo propio)
    print("🔄 Aplicando normalización manual Min-Max...")
    df_feat_norm = normalizacion_min_max_manual(df_feat, features)
    
    # 3. Entrenar y Guardar K-Means
    print("🤖 Entrenando K-Means...")
    try:
        model_km, labels_km, metrics_km = train_clustering_model(df_feat_norm, "kmeans", {"n_clusters": 4})
        path_km = save_model(model_km, "kmeans")
        print(f"   ✅ K-Means guardado en: {path_km}")
        print(f"   📈 Silhouette Score: {metrics_km.get('Silhouette Score', 0.0):.4f}")
    except Exception as e:
        print(f"   ❌ Falló entrenamiento de K-Means: {e}")
        
    # 4. Entrenar y Guardar GMM
    print("🤖 Entrenando Gaussian Mixture Model (GMM)...")
    try:
        model_gmm, labels_gmm, metrics_gmm = train_clustering_model(df_feat_norm, "gmm", {"n_components": 4})
        path_gmm = save_model(model_gmm, "gmm")
        print(f"   ✅ GMM guardado en: {path_gmm}")
        print(f"   📈 Silhouette Score: {metrics_gmm.get('Silhouette Score', 0.0):.4f}")
    except Exception as e:
        print(f"   ❌ Falló entrenamiento de GMM: {e}")
        
    # 5. Entrenar y Guardar DBSCAN
    print("🤖 Entrenando DBSCAN...")
    try:
        model_db, labels_db, metrics_db = train_clustering_model(df_feat_norm, "dbscan", {"eps": 1.5, "min_samples": 5})
        path_db = save_model(model_db, "dbscan")
        print(f"   ✅ DBSCAN guardado en: {path_db}")
        print(f"   📈 Puntos de ruido: {metrics_db.get('Noise points', 0)}")
    except Exception as e:
        print(f"   ❌ Falló entrenamiento de DBSCAN: {e}")
        
    print("\n🎉 ¡Lote de modelos entrenado y actualizado con éxito en la carpeta /models!")

if __name__ == "__main__":
    main()
