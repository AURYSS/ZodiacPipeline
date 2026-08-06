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
from modules.report_generator import generate_pdf_report

# Mapeo robusto global de signos a sus elementos correspondientes
SIGN_TO_ELEMENT = {
    "Aries": "Fuego", "Leo": "Fuego", "Sagitario": "Fuego", "Fuego": "Fuego",
    "Tauro": "Tierra", "Virgo": "Tierra", "Capricornio": "Tierra", "Tierra": "Tierra",
    "Géminis": "Aire", "Libra": "Aire", "Acuario": "Aire", "Aire": "Aire",
    "Cáncer": "Agua", "Escorpio": "Agua", "Piscis": "Agua", "Agua": "Agua"
}

# Configuración de página de Streamlit
st.set_page_config(
    page_title="AI.Studio - Zodiacal Clustering Pipeline",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilización premium personalizada (Zodiac Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* Fondo Cósmico */
    .stApp {
        background: radial-gradient(circle at center, #1b0a31 0%, #080312 100%);
        color: #e2e8f0;
    }
    
    /* Agregar constelaciones sutiles simuladas mediante ruido radial o imagen (opcional, aquí usamos un gradiente profundo) */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 4px),
                          radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 3px),
                          radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 4px);
        background-size: 550px 550px, 350px 350px, 250px 250px;
        background-position: 0 0, 40px 60px, 130px 270px;
        opacity: 0.15;
        z-index: -1;
    }

    h1, h2, h3, .main-title, .section-header {
        font-family: 'Cinzel', serif !important;
    }

    p, span, div {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #d4af37, #f3e5ab, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0px 4px 20px rgba(212, 175, 55, 0.3);
    }
    .sub-title {
        color: #bfa87a;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #d4af37;
        border-bottom: 2px solid rgba(212, 175, 55, 0.2);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        text-shadow: 0px 2px 10px rgba(212, 175, 55, 0.2);
    }
    
    /* Panel de Glassmorphism Cósmico */
    .metric-card {
        background-color: rgba(27, 10, 49, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(212, 175, 55, 0.15);
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Cinzel', serif;
        background: linear-gradient(135deg, #f3e5ab, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #bfa87a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Personalización de Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(8, 3, 18, 0.85) !important;
        border-right: 1px solid rgba(212, 175, 55, 0.15);
    }
    
    /* Botones Dorados */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #4c1d95) !important;
        color: #f3e5ab !important;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9333ea, #5b21b6) !important;
        border-color: #d4af37 !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.4) !important;
        transform: scale(1.02);
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
st.markdown('<div class="sub-title">Extracción de Conocimiento en Base de Datos • Unidad IV: Análisis No Supervisado</div>', unsafe_allow_html=True)

# Menú de Navegación Lateral (7 Pantallas)
st.sidebar.image("static/logo.png", use_container_width=True)
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
st.sidebar.markdown("**Universidad Tecnológica del Norte de Guanajuato**")
st.sidebar.markdown("**Ingeniería en Desarrollo y Gestión de Software**")

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
    st.markdown('<div class="section-header">1. Ingesta y Persistencia</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Suba el archivo CSV con las respuestas Likert (`p1`-`p15`) y respuestas abiertas (`p1a`-`p15a`) para insertarlas en la base de datos PostgreSQL local `zodiac`.")
        uploaded_file = st.file_uploader("Subir dataset en formato .csv", type=["csv"])
        
        if uploaded_file is not None:
            df, msg = load_and_validate_csv(uploaded_file)
            if df is not None:
                st.success(f"¡Carga en memoria exitosa! {msg}")
                if st.button("Guardar e Ingestar", use_container_width=True):
                    try:
                        cant = insertar_encuestas(df)
                        st.success(f"¡Persistencia exitosa! Se insertaron {cant} registros en la base de datos local.")
                        # Recargar de BD para mantener el ID de la BD
                        st.session_state.df_raw = obtener_todas_las_encuestas()
                        st.session_state.df_filtered = st.session_state.df_raw.copy()
                    except Exception as e:
                        if isinstance(e, UnicodeDecodeError):
                            st.error("Error de conexión: El servidor PostgreSQL local rechazó la conexión. Por favor verifica que tu contraseña del usuario 'postgres' sea correcta y que hayas creado la base de datos vacía llamada 'zodiac' en pgAdmin.")
                        else:
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
                    if isinstance(e, UnicodeDecodeError):
                        st.error("Error de conexión: El servidor PostgreSQL local rechazó la conexión. Verifica tu contraseña y que la base de datos 'zodiac' exista.")
                    else:
                        st.error(f"Error de conexión a Postgres: {e}")
            else:
                st.error("No se encontró el dataset en 'data/datos_prueba_zodiac.csv'.")
                
        if st.button("Vaciar Tabla 'encuestas' ", use_container_width=True):
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
    st.markdown('<div class="section-header">2. Consulta Tabular</div>', unsafe_allow_html=True)
    
    # Intentar recargar desde la BD
    try:
        st.session_state.df_raw = obtener_todas_las_encuestas()
    except Exception as e:
        st.error(f"No se pudo consultar PostgreSQL: {e}")
        
    if st.session_state.df_raw is None or st.session_state.df_raw.empty:
        st.warning("La base de datos está vacía. Ingeste datos en la pestaña '1. Ingesta de Datos'.")
    else:
        st.write("Vista de los registros:")
        
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
                params["n_clusters"] = st.slider("Número de Clústeres (K) por Elemento:", 2, 8, 3)
            elif algo == "DBSCAN":
                params["eps"] = st.slider("Epsilon (Eps):", 0.1, 5.0, 0.8, step=0.1)
                params["min_samples"] = st.slider("Mínimo de Muestras (min_samples):", 2, 15, 5)
            elif algo == "Gaussian Mixture Models (GMM)":
                params["n_components"] = st.slider("Número de Componentes:", 2, 8, 3)
                
            normalize = st.checkbox("Normalizar Características mediante Min-Max Manual", value=True)
            
        with col2:
            st.markdown("### Arquitectura de Submodelos Segmentados")
            st.info("💡 **Solución al Sesgo de Nulos**: Se entrenarán 4 submodelos independientes en paralelo (uno por cada elemento). Cada modelo usará únicamente las 3 preguntas correspondientes a su elemento (Fuego: p1-p3, Agua: p4-p6, Aire: p7-p9, Tierra: p10-p12), eliminando la necesidad de imputar valores en las preguntas que el usuario no contestó.")
            
        if st.button("Iniciar Aprendizaje de IA", use_container_width=True):
            algo_key = "kmeans" if algo == "K-Means" else "dbscan" if algo == "DBSCAN" else "gmm"
            
            elementos_vars = {
                "Fuego": ["p1", "p2", "p3"],
                "Agua": ["p4", "p5", "p6"],
                "Aire": ["p7", "p8", "p9"],
                "Tierra": ["p10", "p11", "p12"]
            }
            
            all_metrics = {}
            df_res = df.copy()
            df_res["Cluster"] = -1  # Inicializar
            modelos_paths = {}
            
            # Mapear signos a elementos para filtrar
            df["elemento_temp"] = df["signo"].map(SIGN_TO_ELEMENT)
            for elem, features in elementos_vars.items():
                df_elem_indices = df[df["elemento_temp"] == elem].index
                df_elem = df.loc[df_elem_indices].copy()
                
                if len(df_elem) == 0:
                    continue
                    
                df_feat = df_elem[features].copy()
                if normalize:
                    df_feat = normalizacion_min_max_manual(df_feat, features)
                    
                model, labels, metrics = train_clustering_model(df_feat, algo_key, params)
                
                # Asignar etiquetas a la fila correspondiente
                df_res.loc[df_elem_indices, "Cluster"] = labels
                
                # Guardar modelo
                saved_path = save_model(model, algo_key, elem)
                modelos_paths[elem] = saved_path
                all_metrics[elem] = metrics
                
            st.session_state.last_algorithm = algo_key
            st.session_state.df_results = df_res
            st.session_state.model_metrics = all_metrics
            st.session_state.last_model_path = list(modelos_paths.values())[0] if modelos_paths else None
            st.session_state.modelos_paths_dict = modelos_paths
            
            st.success(f"¡Los 4 submodelos de {algo} fueron entrenados y guardados con éxito!")
            
            st.markdown("### Métricas de Evaluación por Elemento")
            c1, c2, c3, c4 = st.columns(4)
            cols_elem = [c1, c2, c3, c4]
            for idx, (elem, metrics) in enumerate(all_metrics.items()):
                with cols_elem[idx]:
                    st.markdown(f"**Elemento {elem}**")
                    st.metric("Silhouette Score", f"{metrics.get('Silhouette Score', 0.0):.4f}")
                    if "Inertia" in metrics:
                        st.metric("Inercia (Inertia)", f"{metrics.get('Inertia', 0.0):.2f}")
                    elif "Noise points" in metrics:
                        st.metric("Puntos de ruido", f"{metrics.get('Noise points', 0)}")
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
        
        # Selector de Elemento para Visualizar
        st.markdown("### Visualización por Elemento Astrológico")
        elem_visualizar = st.selectbox(
            "Seleccione el elemento para visualizar su agrupamiento y análisis independiente:",
            ["Fuego", "Agua", "Aire", "Tierra"]
        )
        
        # Filtrar datos de este elemento robustamente
        df_res["elemento_temp"] = df_res["signo"].map(SIGN_TO_ELEMENT)
        df_elem = df_res[df_res["elemento_temp"] == elem_visualizar].copy()
        
        if df_elem.empty:
            st.warning(f"No hay registros cargados para el elemento {elem_visualizar}.")
        else:
            elementos_vars = {
                "Fuego": ["p1", "p2", "p3"],
                "Agua": ["p4", "p5", "p6"],
                "Aire": ["p7", "p8", "p9"],
                "Tierra": ["p10", "p11", "p12"]
            }
            features = elementos_vars[elem_visualizar]
            
            df_feat = df_elem[features].copy()
            df_pca, variance = apply_pca_reduction(df_feat, n_components=3)
            
            st.write(f"Visualización de los sub-clústeres del elemento **{elem_visualizar}** entrenados en base a: `{features}`")
            st.write(f"Varianza explicada acumulada (PC1 + PC2 + PC3): **{sum(variance)*100:.2f}%**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_pca_2d(df_pca, df_elem["Cluster"], hover_data=df_elem[["id", "edad", "genero"]]), use_container_width=True)
            with col2:
                st.plotly_chart(plot_pca_3d(df_pca, df_elem["Cluster"], hover_data=df_elem[["id", "edad", "genero"]]), use_container_width=True)
                
            # NLP Cualitativo específico del elemento
            st.markdown(f"### Análisis Cualitativo NLP: Respuestas de {elem_visualizar}")
            
            # Columnas abiertas específicas para el elemento
            open_cols_mapping = {
                "Fuego": ["p1a", "p2a", "p3a"],
                "Agua": ["p4a", "p5a", "p6a"],
                "Aire": ["p7a", "p8a", "p9a"],
                "Tierra": ["p10a", "p11a", "p12a"]
            }
            nlp_cols = open_cols_mapping[elem_visualizar]
            
            df_sent = analizar_sentimientos_dataset(df_elem, nlp_cols)
            df_elem["Polaridad_Sentimiento_Promedio"] = df_sent["sentimiento_promedio"]
            
            st.write("Polaridad promedio de sentimiento por participante (1.0 = Positivo, -1.0 = Negativo):")
            st.dataframe(df_elem[["id", "Cluster", "Polaridad_Sentimiento_Promedio"] + nlp_cols].head(15), use_container_width=True)
            
            st.write(f"Términos más recurrentes descubiertos en las respuestas abiertas de **{elem_visualizar}** (TF-IDF):")
            df_tfidf = obtener_terminos_frecuentes_tfidf(df_elem, nlp_cols)
            st.dataframe(df_tfidf, use_container_width=True)

# ==============================================================================
# PANTALLA 7: ZONA DE DESCARGAS
# ==============================================================================
elif pantalla == "7. Zona de Descargas":
    st.markdown('<div class="section-header">7. Exportación e Informes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Descargar Resultados (Excel / CSV)")
        if st.session_state.df_results is not None:
            df_res = st.session_state.df_results
            
            tipo_descarga = st.radio("Seleccione el formato y tipo de datos:", 
                                     ["CSV Completo", "Excel Cuantitativo (Numérico)", "Excel Cualitativo (Texto/Categorías)"])
            
            if tipo_descarga == "CSV Completo":
                csv_res = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Dataset Completo (CSV)",
                    data=csv_res,
                    file_name="resultados_zodiacal_clustering.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    if "Cuantitativo" in tipo_descarga:
                        cols_cuant = ["id", "edad"] + [f"p{i}" for i in range(1, 16)]
                        if "Cluster" in df_res.columns:
                            cols_cuant.append("Cluster")
                        cols_export = [c for c in cols_cuant if c in df_res.columns]
                        df_res[cols_export].to_excel(writer, index=False, sheet_name="Cuantitativo")
                        file_name = "resultados_cuantitativos.xlsx"
                    else:
                        cols_cual = ["id", "genero", "signo"] + [f"p{i}a" for i in range(1, 16)]
                        if "Polaridad_Sentimiento_Promedio" in df_res.columns:
                            cols_cual.append("Polaridad_Sentimiento_Promedio")
                        if "Cluster" in df_res.columns:
                            cols_cual.append("Cluster")
                        cols_export = [c for c in cols_cual if c in df_res.columns]
                        df_res[cols_export].to_excel(writer, index=False, sheet_name="Cualitativo")
                        file_name = "resultados_cualitativos.xlsx"
                        
                excel_data = output.getvalue()
                st.download_button(
                    label=f"Descargar {tipo_descarga} (Excel)",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("No se han generado resultados aún.")
            
        st.markdown("### Descargar Modelo de Entrenamiento (.pkl)")
        if "modelos_paths_dict" in st.session_state and st.session_state.modelos_paths_dict:
            # Selector de elemento para descargar el modelo
            elem_descargar = st.selectbox(
                "Seleccione el modelo del elemento a descargar (.pkl):",
                list(st.session_state.modelos_paths_dict.keys())
            )
            path_modelo = st.session_state.modelos_paths_dict[elem_descargar]
            if os.path.exists(path_modelo):
                with open(path_modelo, "rb") as f:
                    model_bytes = f.read()
                st.download_button(
                    label=f"Descargar Modelo de {elem_descargar} (.pkl)",
                    data=model_bytes,
                    file_name=os.path.basename(path_modelo),
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("No hay ningún modelo entrenado guardado.")
            
    with col2:
        st.markdown("### Reporte Estadístico Ejecutivo")
        st.write("Descarga un informe profesional en PDF listo para presentar.")

        st.write("Descarga un informe profesional en PDF con análisis detallado.")
        
        if st.session_state.df_results is not None:
            df = st.session_state.df_results
            metrics = st.session_state.model_metrics
            algo = st.session_state.last_algorithm
            
            cluster_counts = df["Cluster"].value_counts().to_dict()
            
            # Generar gráficos en PNG usando Kaleido para inyectar en el PDF
            graphs_bytes = []
            try:
                features = ["p1", "p2", "p3"] # Fuego por defecto
                df_feat = df[df["signo"].map(SIGN_TO_ELEMENT) == "Fuego"][features].copy()
                if not df_feat.empty:
                    df_pca, _ = apply_pca_reduction(df_feat, n_components=2)
                    labels = df[df["signo"].map(SIGN_TO_ELEMENT) == "Fuego"]["Cluster"]
                    
                    fig1 = plot_pca_2d(df_pca, labels)
                    graphs_bytes.append(fig1.to_image(format="png", width=800, height=600))
            except Exception as e:
                st.warning("⚠️ Nota: Las gráficas de dispersión no se incluirán en el PDF generado debido a la falta de Google Chrome / dependencias de Kaleido en este entorno local. El reporte con tablas de métricas y frecuencias se descargará con normalidad.")
            
            # Obtener el primer set de métricas si es un diccionario
            pdf_metrics = list(metrics.values())[0] if isinstance(metrics, dict) else metrics
            
            pdf_bytes = bytes(generate_pdf_report(algo, pdf_metrics, cluster_counts, graphs_bytes))
            
            st.download_button(
                label="📥 Descargar Reporte (PDF)",
                data=pdf_bytes,
                file_name="reporte_ejecutivo_zodiacal.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("Entrene el modelo para poder visualizar el reporte.")
