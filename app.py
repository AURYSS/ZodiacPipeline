import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import joblib

# Importar módulos locales
from modules.data_loader import load_and_validate_csv
from modules.database import (
    init_db,
    insertar_encuestas,
    obtener_todas_las_encuestas,
    limpiar_base_de_datos
)
from modules.statistics import (
    calcular_promedio_manual,
    calcular_moda_manual,
    calcular_varianza_manual,
    calcular_desviacion_estandar_manual,
    normalizacion_min_max_manual,
    calcular_promedios_dimensiones
)
from modules.clustering import train_clustering_model, apply_pca_reduction, save_model, MODELS_DIR
from modules.nlp import analizar_sentimientos_dataset, obtener_terminos_frecuentes_tfidf
from modules.visualizations import (
    plot_pca_2d,
    plot_pca_3d,
    plot_distribucion_signos_cluster,
    plot_perfil_clusters,
    plot_distribucion_signo,
    plot_matriz_correlacion
)

# Configuración de página de Streamlit
st.set_page_config(
    page_title="AI.Studio - Zodiacal Clustering Pipeline",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilización premium personalizada
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estados de sesión
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "last_model_path" not in st.session_state:
    st.session_state.last_model_path = None
if "last_algorithm" not in st.session_state:
    st.session_state.last_algorithm = None
if "features_used" not in st.session_state:
    st.session_state.features_used = []

# Título y Subtítulo
st.markdown('<div class="main-title">🔮 Zodiacal Clustering Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unidad IV: Análisis No Supervisado • Base de Datos local PostgreSQL (zodiac)</div>', unsafe_allow_html=True)

# Menú de Navegación Lateral (7 Pantallas)
st.sidebar.image("https://img.icons8.com/nolan/96/crystal-ball.png", width=80)
st.sidebar.markdown("### Navegación del Sistema")
pantalla = st.sidebar.radio(
    "Seleccione una pantalla:",
    [
        "1. Ingesta de Datos",
        "2. Visualizar Datos",
        "3. Filtros y Exportación",
        "4. Estadísticas Básicas",
        "5. Entrenar Modelo",
        "6. Resultados de IA & NLP",
        "7. Zona de Descargas"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Base de Datos:** PostgreSQL (zodiac)")
st.sidebar.markdown("**Estudiante:** Ing. en Desarrollo y Gestión de Software")

# Intentar cargar datos de PostgreSQL si existen al inicio
try:
    df_db = obtener_todas_las_encuestas()
    if not df_db.empty and st.session_state.df_raw is None:
        st.session_state.df_raw = df_db
        st.session_state.df_filtered = df_db.copy()
except Exception as e:
    pass

# ==============================================================================
# PANTALLA 1: INGESTA DE DATOS
# ==============================================================================
if pantalla == "1. Ingesta de Datos":
    st.markdown('<div class="section-header">1. Ingesta y Persistencia en PostgreSQL</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Suba el archivo CSV con las respuestas para insertarlas en la base de datos PostgreSQL local `zodiac`.")
        uploaded_file = st.file_uploader("Subir dataset en formato .csv", type=["csv"])
        
        if uploaded_file is not None:
            df, msg = load_and_validate_csv(uploaded_file)
            if df is not None:
                st.success(f"¡Carga en memoria exitosa! {msg}")
                if st.button("Guardar e Ingestar en PostgreSQL", use_container_width=True):
                    try:
                        cant = insertar_encuestas(df)
                        st.success(f"¡Persistencia exitosa! Se insertaron {cant} registros en la base de datos local.")
                        # Recargar de BD para mantener el ID de la BD
                        st.session_state.df_raw = obtener_todas_las_encuestas()
                        st.session_state.df_filtered = st.session_state.df_raw.copy()
                    except Exception as e:
                        st.error(f"Error de conexión a Postgres: {e}")
            else:
                st.error(f"Error de Validación: {msg}")
                
    with col2:
        st.markdown("### Acciones Demo & BD")
        if st.button("Cargar y Guardar Dataset de Prueba (5000 registros)", use_container_width=True):
            path_demo = os.path.join(os.path.dirname(__file__), "data", "datos_prueba_zodiac.csv")
            if os.path.exists(path_demo):
                df_demo = pd.read_csv(path_demo)
                try:
                    cant = insertar_encuestas(df_demo)
                    st.success(f"¡Éxito! {cant} registros insertados en Postgres.")
                    st.session_state.df_raw = obtener_todas_las_encuestas()
                    st.session_state.df_filtered = st.session_state.df_raw.copy()
                except Exception as e:
                    st.error(f"Error al conectar con Postgres: {e}")
            else:
                st.error("No se encontró el dataset en 'data/datos_prueba_zodiac.csv'.")
                
        if st.button("Vaciar Tabla 'encuestas' en PostgreSQL", use_container_width=True):
            try:
                limpiar_base_de_datos()
                st.session_state.df_raw = None
                st.session_state.df_filtered = None
                st.session_state.df_results = None
                st.success("Tabla vaciada con éxito en PostgreSQL.")
            except Exception as e:
                st.error(f"Error de base de datos: {e}")
                
    if st.session_state.df_raw is not None:
        st.markdown("### Estado de la Base de Datos")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total en Postgres", f"{len(st.session_state.df_raw)} registros")
        with c2:
            st.metric("Columnas Likert", "p1 - p15")
        with c3:
            st.metric("Columnas Abiertas", "p1a - p15a")

# ==============================================================================
# PANTALLA 2: VISUALIZAR DATOS
# ==============================================================================
elif pantalla == "2. Visualizar Datos":
    st.markdown('<div class="section-header">2. Consulta Tabular de PostgreSQL</div>', unsafe_allow_html=True)
    
    # Intentar recargar desde la BD
    try:
        st.session_state.df_raw = obtener_todas_las_encuestas()
    except Exception as e:
        st.error(f"No se pudo consultar PostgreSQL: {e}")
        
    if st.session_state.df_raw is None or st.session_state.df_raw.empty:
        st.warning("La base de datos está vacía. Ingeste datos en la pestaña '1. Ingesta de Datos'.")
    else:
        st.write("Vista de los registros directo de la base de datos PostgreSQL:")
        
        records_per_page = st.selectbox("Registros por página", [10, 25, 50, 100], index=0)
        total_len = len(st.session_state.df_raw)
        pages = max(1, int(np.ceil(total_len / records_per_page)))
        page_num = st.number_input("Página", min_value=1, max_value=pages, value=1)
        
        start_idx = (page_num - 1) * records_per_page
        end_idx = min(start_idx + records_per_page, total_len)
        
        st.write(f"Mostrando registros del {start_idx + 1} al {end_idx} de un total de {total_len}.")
        st.dataframe(st.session_state.df_raw.iloc[start_idx:end_idx], use_container_width=True)

# ==============================================================================
# PANTALLA 3: FILTROS Y EXPORTACIÓN
# ==============================================================================
elif pantalla == "3. Filtros y Exportación":
    st.markdown('<div class="section-header">3. Segmentación y Filtros de Información</div>', unsafe_allow_html=True)
    
    try:
        st.session_state.df_raw = obtener_todas_las_encuestas()
    except:
        pass
        
    if st.session_state.df_raw is None or st.session_state.df_raw.empty:
        st.warning("La base de datos está vacía. Cargue datos antes de filtrar.")
    else:
        df = st.session_state.df_raw.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            min_age = int(df["edad"].min())
            max_age = int(df["edad"].max())
            age_range = st.slider("Rango de Edad:", min_age, max_age, (min_age, max_age))
            
            generos_disponibles = ["Todos"] + df["genero"].unique().tolist()
            genero_selected = st.selectbox("Seleccione Género:", generos_disponibles)
            
        with col2:
            signos_disponibles = ["Todos"] + df["signo"].unique().tolist()
            signo_selected = st.selectbox("Seleccione Signo Zodiacal:", signos_disponibles)
            
        df_filtered = df[(df["edad"] >= age_range[0]) & (df["edad"] <= age_range[1])]
        
        if genero_selected != "Todos":
            df_filtered = df_filtered[df_filtered["genero"] == genero_selected]
            
        if signo_selected != "Todos":
            df_filtered = df_filtered[df_filtered["signo"] == signo_selected]
            
        st.session_state.df_filtered = df_filtered
        
        st.markdown(f"#### Resultados Filtrados: {len(df_filtered)} registros encontrados.")
        st.dataframe(df_filtered, use_container_width=True)
        
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Datos Filtrados (CSV)",
            data=csv_data,
            file_name="datos_filtrados_zodiac.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==============================================================================
# PANTALLA 4: ESTADÍSTICAS BÁSICAS
# ==============================================================================
elif pantalla == "4. Estadísticas Básicas":
    st.markdown('<div class="section-header">4. Estadísticas Descriptivas (Algoritmos Propios)</div>', unsafe_allow_html=True)
    
    if st.session_state.df_filtered is None or len(st.session_state.df_filtered) == 0:
        st.warning("No hay registros filtrados o la base de datos está vacía.")
    else:
        df = st.session_state.df_filtered.copy()
        likert_cols = [f"p{i}" for i in range(1, 16)]
        
        st.markdown("### Métricas de Rúbrica de Evaluación por Pregunta (Likert)")
        st.write("Calculados usando **funciones manuales programadas paso a paso** (Rúbrica de evaluación):")
        
        stat_rows = []
        for col in likert_cols:
            val_col = df[col].tolist()
            promedio = calcular_promedio_manual(val_col)
            moda = calcular_moda_manual(val_col)
            varianza = calcular_varianza_manual(val_col)
            std_dev = calcular_desviacion_estandar_manual(val_col)
            stat_rows.append({
                "Pregunta": col,
                "Promedio (Manual)": round(promedio, 3),
                "Moda (Manual)": moda,
                "Varianza (Manual)": round(varianza, 3),
                "Desviación Estándar (Manual)": round(std_dev, 3)
            })
            
        st.table(pd.DataFrame(stat_rows))
        
        st.markdown("### Visualización de Variables Demográficas")
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.plotly_chart(plot_distribucion_signo(df), use_container_width=True)
        with col_viz2:
            st.plotly_chart(plot_matriz_correlacion(df[likert_cols]), use_container_width=True)

# ==============================================================================
# PANTALLA 5: ENTRENAR MODELO
# ==============================================================================
elif pantalla == "5. Entrenar Modelo":
    st.markdown('<div class="section-header">5. Configuración y Entrenamiento del Modelo</div>', unsafe_allow_html=True)
    
    try:
        st.session_state.df_raw = obtener_todas_las_encuestas()
    except:
        pass
        
    if st.session_state.df_raw is None or st.session_state.df_raw.empty:
        st.warning("La base de datos está vacía. Ingeste datos primero.")
    else:
        df = st.session_state.df_raw.copy()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Configuración de Algoritmo")
            algo = st.selectbox("Algoritmo de Agrupamiento:", ["K-Means", "DBSCAN", "Gaussian Mixture Models (GMM)"])
            
            params = {}
            if algo == "K-Means":
                params["n_clusters"] = st.slider("Número de Clústeres (K):", 2, 12, 4)
            elif algo == "DBSCAN":
                params["eps"] = st.slider("Epsilon (Eps):", 0.1, 5.0, 1.5, step=0.1)
                params["min_samples"] = st.slider("Mínimo de Muestras (min_samples):", 2, 15, 5)
            elif algo == "Gaussian Mixture Models (GMM)":
                params["n_components"] = st.slider("Número de Componentes:", 2, 12, 4)
                
            normalize = st.checkbox("Normalizar Características mediante Min-Max Manual", value=True)
            
        with col2:
            st.markdown("### Selección de Características (Features)")
            st.write("Seleccione qué preguntas Likert usar para el entrenamiento (El signo zodiacal está excluido):")
            
            likert_cols = [f"p{i}" for i in range(1, 16)]
            selected_features = []
            for col in likert_cols:
                if st.checkbox(f"Pregunta {col} (Likert)", value=True, key=f"feat_{col}"):
                    selected_features.append(col)
                    
        if st.button("Iniciar Aprendizaje de IA", use_container_width=True):
            if len(selected_features) == 0:
                st.error("Debe seleccionar al menos una característica numérica para entrenar.")
            else:
                df_feat = df[selected_features].copy()
                
                if normalize:
                    df_feat = normalizacion_min_max_manual(df_feat, selected_features)
                    
                algo_key = "kmeans" if algo == "K-Means" else "dbscan" if algo == "DBSCAN" else "gmm"
                model, labels, metrics = train_clustering_model(df_feat, algo_key, params)
                
                st.session_state.features_used = selected_features
                st.session_state.last_algorithm = algo_key
                st.session_state.model_metrics = metrics
                
                df_res = df.copy()
                df_res["Cluster"] = labels
                st.session_state.df_results = df_res
                
                saved_path = save_model(model, algo_key)
                st.session_state.last_model_path = saved_path
                
                st.success(f"¡Modelo {algo} entrenado y guardado con éxito!")
                
                st.markdown("### Métricas de Evaluación del Modelo")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Silhouette Score (Silueta)", f"{metrics.get('Silhouette Score', 0.0):.4f}")
                with m2:
                    if "Inertia" in metrics:
                        st.metric("Inertia (Inercia)", f"{metrics.get('Inertia', 0.0):.2f}")
                    elif "Noise points" in metrics:
                        st.metric("Puntos de ruido (Outliers)", f"{metrics.get('Noise points', 0)}")
                    elif "BIC" in metrics:
                        st.metric("BIC Score", f"{metrics.get('BIC', 0.0):.2f}")

# ==============================================================================
# PANTALLA 6: RESULTADOS DE IA & NLP
# ==============================================================================
elif pantalla == "6. Resultados de IA & NLP":
    st.markdown('<div class="section-header">6. Resultados y Análisis Cualitativo NLP</div>', unsafe_allow_html=True)
    
    if st.session_state.df_results is None:
        st.warning("No hay resultados de entrenamiento. Por favor entrene un modelo primero.")
    else:
        df_res = st.session_state.df_results.copy()
        features = st.session_state.features_used
        
        df_feat = df_res[features].copy()
        df_pca, variance = apply_pca_reduction(df_feat, n_components=3)
        
        st.markdown("### Visualización PCA 2D y 3D")
        st.write(f"Varianza explicada acumulada (PC1 + PC2 + PC3): **{sum(variance)*100:.2f}%**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_pca_2d(df_pca, df_res["Cluster"], hover_data=df_res[["id", "edad", "signo", "genero"]]), use_container_width=True)
        with col2:
            st.plotly_chart(plot_pca_3d(df_pca, df_res["Cluster"], hover_data=df_res[["id", "edad", "signo", "genero"]]), use_container_width=True)
            
        st.markdown("### Validación: Comparativa de Clústeres contra Signo Zodiacal")
        st.plotly_chart(plot_distribucion_signos_cluster(df_res), use_container_width=True)
        
        st.markdown("### Análisis Cualitativo NLP de Preguntas Abiertas")
        open_cols = [f"p{i}a" for i in range(1, 16)]
        selected_nlp_cols = st.multiselect("Seleccione preguntas abiertas para analizar:", open_cols, default=["p1a", "p2a", "p3a"])
        
        if selected_nlp_cols:
            df_sent = analizar_sentimientos_dataset(df_res, selected_nlp_cols)
            df_res["Polaridad_Sentimiento_Promedio"] = df_sent["sentimiento_promedio"]
            
            st.write("Polaridad promedio de sentimiento asignada por persona (1.0 = Positivo, -1.0 = Negativo):")
            st.dataframe(df_res[["id", "signo", "Cluster", "Polaridad_Sentimiento_Promedio"] + selected_nlp_cols].head(15), use_container_width=True)
            
            st.write("Términos más significativos descubiertos en respuestas abiertas (TF-IDF):")
            df_tfidf = obtener_terminos_frecuentes_tfidf(df_res, selected_nlp_cols)
            st.dataframe(df_tfidf, use_container_width=True)

# ==============================================================================
# PANTALLA 7: ZONA DE DESCARGAS
# ==============================================================================
elif pantalla == "7. Zona de Descargas":
    st.markdown('<div class="section-header">7. Exportación e Informes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Descargar Resultados en CSV")
        if st.session_state.df_results is not None:
            csv_res = st.session_state.df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Dataset de Resultados (CSV)",
                data=csv_res,
                file_name="resultados_zodiacal_clustering.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No se han generado resultados aún.")
            
        st.markdown("### Descargar Modelo de Entrenamiento (.pkl)")
        if st.session_state.last_model_path is not None and os.path.exists(st.session_state.last_model_path):
            with open(st.session_state.last_model_path, "rb") as f:
                model_bytes = f.read()
            st.download_button(
                label="Descargar Modelo Entrenado (.pkl)",
                data=model_bytes,
                file_name=os.path.basename(st.session_state.last_model_path),
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            st.info("No hay ningún modelo entrenado guardado.")
            
    with col2:
        st.markdown("### Reporte Estadístico Resumido")
        if st.session_state.df_results is not None:
            df = st.session_state.df_results
            metrics = st.session_state.model_metrics
            
            report = f"""======================================================
REPORTE ESTADÍSTICO DE PIPELINE ZODIACAL CLUSTERING
======================================================
Algoritmo Utilizado: {st.session_state.last_algorithm.upper()}
Métricas de Rúbrica:
- Silhouette Score: {metrics.get('Silhouette Score', 0.0):.4f}
"""
            if "Inertia" in metrics:
                report += f"- Inertia: {metrics.get('Inertia', 0.0):.2f}\n"
            elif "Noise points" in metrics:
                report += f"- Puntos de Ruido: {metrics.get('Noise points', 0)}\n"
            
            report += f"\nResumen por Clúster:\n"
            cluster_counts = df["Cluster"].value_counts().to_dict()
            for clust, count in cluster_counts.items():
                report += f"- Clúster {clust}: {count} individuos\n"
                
            st.text_area("Vista previa del reporte:", report, height=200)
            
            st.download_button(
                label="Descargar Reporte Completo (TXT)",
                data=report.encode('utf-8'),
                file_name="reporte_zodiacal_clustering.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Entrene el modelo para poder visualizar el reporte.")
