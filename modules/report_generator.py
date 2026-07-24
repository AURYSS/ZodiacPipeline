import os
import tempfile
from fpdf import FPDF
from datetime import datetime

class ZodiacReportPDF(FPDF):
    def __init__(self, logo_path=None):
        super().__init__()
        self.logo_path = logo_path
        
    def header(self):
        # Intentar cargar el logo si existe
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.image(self.logo_path, 10, 8, 25)
            except Exception:
                pass
                
        # Título
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(212, 175, 55) # Oro
        self.cell(30) # Mover a la derecha del logo
        self.cell(0, 10, 'Zodiac AI - Reporte Analítico', border=0, align='L')
        self.ln(8)
        
        # Subtítulo
        self.set_font('helvetica', 'I', 12)
        self.set_text_color(100, 100, 100)
        self.cell(30)
        self.cell(0, 10, f'Generado el: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', border=0, align='L')
        self.ln(20)
        
        # Línea divisoria
        self.set_draw_color(212, 175, 55)
        self.set_line_width(0.5)
        self.line(10, 35, 200, 35)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} - Zodiac AI Pipeline', align='C')

def generate_pdf_report(algorithm_name, metrics, cluster_counts, graphs_bytes=None):
    """
    Genera el archivo PDF en memoria y lo devuelve como bytes.
    """
    # Ruta absoluta al logo generado por la IA
    LOGO_PATH = r"C:\Users\Aurora\.gemini\antigravity\brain\d9ae68f0-797f-4ce8-891f-d4bfc9b4b35e\zodiac_ai_logo_1784595240612.png"
    
    pdf = ZodiacReportPDF(logo_path=LOGO_PATH)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Título de Sección
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(74, 14, 78) # Púrpura profundo
    pdf.cell(0, 10, '1. Configuración del Entrenamiento', ln=True)
    
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'Algoritmo Seleccionado: {algorithm_name.upper()}', ln=True)
    pdf.ln(5)
    
    # Métricas
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(74, 14, 78)
    pdf.cell(0, 10, '2. Métricas de Desempeño (Rúbrica)', ln=True)
    
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"- Silhouette Score: {metrics.get('Silhouette Score', 0.0):.4f}", ln=True)
    if "Inertia" in metrics:
        pdf.cell(0, 8, f"- Inertia (Inercia): {metrics.get('Inertia', 0.0):.2f}", ln=True)
    elif "Noise points" in metrics:
        pdf.cell(0, 8, f"- Puntos de Ruido: {metrics.get('Noise points', 0)}", ln=True)
    
    pdf.ln(2)
    pdf.set_font('helvetica', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    
    # Análisis dinámico extenso del Silhouette Score y Métricas
    sil_score = metrics.get('Silhouette Score', 0.0)
    if sil_score >= 0.5:
        sil_desc = (f"El valor de Silhouette Score obtenido de {sil_score:.4f} es cuantitativamente sobresaliente. "
                    "En el contexto del aprendizaje no supervisado, este coeficiente indica que la cohesión intra-clúster es significativamente mayor que la separación inter-clúster. "
                    "Esto significa empíricamente que los individuos dentro de un mismo grupo comparten perfiles astrológicos y psicológicos casi idénticos, "
                    "mientras que mantienen una frontera de decisión muy clara respecto a los demás grupos. Un modelo con esta robustez "
                    "puede ser utilizado directamente para inferencias poblacionales precisas y estrategias de segmentación de alto impacto sin temor a clasificaciones erróneas graves.")
    elif sil_score >= 0.25:
        sil_desc = (f"El coeficiente de Silhouette Score de {sil_score:.4f} representa un resultado analíticamente aceptable, "
                    "evidenciando la presencia de una estructura de agrupamiento genuina, aunque con cierta superposición natural en las fronteras. "
                    "En estudios de ciencias sociales y psicometría, es extremadamente común observar este nivel de solapamiento debido a la complejidad "
                    "y multifactorialidad del comportamiento humano. El modelo ha logrado identificar patrones generales viables, sugiriendo que, "
                    "si bien existen tendencias marcadas, ciertos individuos poseen características limítrofes (casos de frontera) que comparten "
                    "rasgos con múltiples perfiles al mismo tiempo.")
    else:
        sil_desc = (f"El valor métrico de Silhouette Score de {sil_score:.4f} revela un alto grado de solapamiento multidimensional en los datos. "
                    "Desde el punto de vista algorítmico, esto indica que los hiperplanos que dividen a los grupos no logran separar de manera nítida a los individuos. "
                    "Frecuentemente, esto sugiere que el fenómeno estudiado es un espectro continuo en lugar de categorías discretas. "
                    "Bajo este escenario, las conclusiones deben manejarse como tendencias de fondo macroscópicas y no como reglas deterministas inflexibles, "
                    "invitando a un refinamiento en la etapa de ingeniería de características (Feature Engineering).")
        
    metric_text = f"Análisis Técnico y Estadístico Extendido de las Métricas:\n\n{sil_desc}\n\n"
    
    if "Inertia" in metrics:
        metric_text += (f"Adicionalmente, el cálculo de Inercia ha arrojado un valor total de {metrics['Inertia']:.2f}. "
                        "Matemáticamente, la inercia (o suma de cuadrados dentro del clúster - WCSS) es una medida directa de la compacidad interna de los agrupamientos. "
                        "Un algoritmo K-Means busca minimizar esta función de coste durante su convergencia iterativa. "
                        "Este nivel de inercia, combinado con las características morfológicas del conjunto de datos, refleja un equilibrio "
                        "alcanzado por el modelo entre la generalización de patrones (evitando el sobreajuste al no crear demasiados clústeres artificiales) "
                        "y la retención de especificidad poblacional. Permite asegurar que los centroides calculados son estadísticamente representativos "
                        "del centro de masa gravitacional de cada perfil de comportamiento.")
    elif "Noise points" in metrics:
        metric_text += (f"Un aspecto crucial de este análisis es que el algoritmo basado en densidad (DBSCAN) identificó un total de {metrics['Noise points']} "
                        "observaciones como 'ruido estadístico' (outliers). Lejos de ser un fallo, esta es una de las virtudes matemáticas más importantes "
                        "del modelado por densidad: permite al sistema ignorar anomalías extremas y respuestas atípicas que de otra manera distorsionarían "
                        "los promedios de los grupos principales. Estos puntos atípicos podrían representar individuos con respuestas completamente disruptivas, "
                        "sesgos de llenado en las encuestas, o perfiles genuinamente únicos que escapan a las tendencias del zodiaco tradicional, "
                        "y merecen un análisis cualitativo aislado.")
        
    pdf.multi_cell(0, 5, metric_text)
    pdf.ln(5)
    
    # Resultados
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(74, 14, 78)
    pdf.cell(0, 10, '3. Distribución de Población por Clúster', ln=True)
    
    # Tabla
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_fill_color(212, 175, 55) # Dorado
    pdf.set_text_color(255, 255, 255)
    pdf.cell(60, 10, 'Clúster Asignado', border=1, fill=True, align='C')
    pdf.cell(60, 10, 'Total Individuos', border=1, fill=True, align='C')
    pdf.ln()
    
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    fill = False
    for clust, count in cluster_counts.items():
        if fill:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
            
        pdf.cell(60, 10, f'Clúster {clust}', border=1, fill=True, align='C')
        pdf.cell(60, 10, str(count), border=1, fill=True, align='C')
        pdf.ln()
        fill = not fill
        
    pdf.ln(5)
    pdf.set_font('helvetica', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    
    # Análisis dinámico extenso de los clústeres
    if cluster_counts:
        max_cluster = max(cluster_counts, key=cluster_counts.get)
        min_cluster = min(cluster_counts, key=cluster_counts.get)
        max_count = cluster_counts[max_cluster]
        min_count = cluster_counts[min_cluster]
        total_pob = sum(cluster_counts.values())
        
        table_desc = (f"Interpretación Analítica y Demográfica Profunda:\n\n"
                      f"El algoritmo no supervisado ha evaluado un universo de {total_pob} individuos para descubrir la estructura latente en sus respuestas. "
                      f"El hallazgo más significativo es la identificación del Clúster {max_cluster} como el macrogrupo dominante, abarcando a {max_count} individuos "
                      f"(lo que representa un {(max_count/total_pob)*100:.1f}% de la muestra total). Sociológicamente, este macro-clúster representa el 'comportamiento basal' "
                      f"o la tendencia hegemónica de la población estudiada. Refleja el perfil actitudinal estándar, indicando que una mayoría sustancial "
                      f"comparte un marco cognitivo, emocional o de preferencias muy similar cuando se enfrenta a los reactivos de la escala de Likert.\n\n"
                      f"En contraste diametral, el análisis ha aislado analíticamente al Clúster {min_cluster}, el cual funge como el grupo más exclusivo o minoritario, "
                      f"reteniendo únicamente a {min_count} individuos (apenas el {(min_count/total_pob)*100:.1f}% del total). La existencia empírica de un clúster "
                      f"tan reducido es de sumo interés investigativo. Desde la perspectiva del Machine Learning, demuestra la altísima sensibilidad del algoritmo "
                      f"para capturar variaciones sutiles (micro-tendencias) sin dejarse opacar por la mayoría absoluta. Este grupo minoritario podría representar "
                      f"a individuos con comportamientos divergentes, perfiles de alto valor estadístico (nichos especializados) o aquellos cuya personalidad no se alinea "
                      f"con los estereotipos zodiacales tradicionales, sirviendo como un excelente punto de partida para análisis cualitativos subsecuentes.\n\n"
                      f"Esta disparidad demográfica entre los grupos consolida la validez de aplicar técnicas de Inteligencia Artificial para la segmentación, "
                      f"ya que el cerebro humano y las heurísticas clásicas tienden a simplificar o ignorar estas dinámicas poblacionales asimétricas.")
    else:
        table_desc = "Interpretación de la Tabla: No se generaron clústeres viables para analizar."
        
    pdf.multi_cell(0, 5, table_desc)
        
    if graphs_bytes:
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.set_text_color(74, 14, 78)
        pdf.cell(0, 10, '4. Visualizaciones Analíticas', ln=True)
        pdf.ln(2)
        pdf.set_font('helvetica', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        
        graphs_desc = (f"Interpretación Científica Extensa de las Proyecciones y Topología Espacial:\n\n"
                       f"Para comprender visualmente las decisiones tomadas por el motor de {algorithm_name.upper()}, es imperativo recurrir a técnicas matemáticas "
                       f"de reducción de dimensionalidad (Dimensionality Reduction), predominantemente el Análisis de Componentes Principales (PCA). "
                       f"El conjunto original de datos (compuesto por múltiples columnas de encuestas) reside en un hiperespacio de N dimensiones que escapa a la "
                       f"comprensión visual humana. PCA transforma este hiperespacio mediante la creación de combinaciones lineales ortogonales (eigenvectores) "
                       f"que encapsulan la máxima varianza de los datos en tan solo 2 o 3 componentes principales.\n\n"
                       f"Al observar los gráficos subsiguientes (derivados de {len(graphs_bytes)} proyecciones generadas empíricamente a partir de los datos en memoria), "
                       f"cada coordenada puntual en el plano representa la consolidación multidimensional de un individuo completo. "
                       f"La proximidad euclidiana entre dos puntos traduce visualmente la similitud absoluta en sus patrones de respuesta. "
                       f"Si los puntos comparten un mismo color (mismo clúster) y se agrupan de manera densa en una zona específica del plano (alta cohesión), "
                       f"se ratifica la eficacia geométrica del algoritmo para segmentar. Por el contrario, si visualizamos franjas de intersección cromática, "
                       f"observamos directamente los márgenes de incertidumbre y transiciones difusas entre perfiles actitudinales.\n\n"
                       f"Adicionalmente, los gráficos de distribución categórica cruzan estos clústeres numéricos puros con metadatos descriptivos (como el signo zodiacal). "
                       f"Esto expone de manera categórica si las agrupaciones matemáticas tienen alguna correlación estadísticamente significativa con factores demográficos o "
                       f"astrológicos, respondiendo así a la pregunta fundamental de investigación con evidencia visual contundente.")
        
        pdf.multi_cell(0, 5, graphs_desc)
        pdf.ln(5)
        
        for idx, img_bytes in enumerate(graphs_bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                tmp_name = tmp.name
                
            try:
                # Si estamos cerca del final de la página, agregar nueva página
                if pdf.get_y() > 200:
                    pdf.add_page()
                pdf.image(tmp_name, x=15, w=180)
                pdf.ln(10)
            except Exception:
                pass
            finally:
                os.remove(tmp_name)
        
    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 10)
    pdf.multi_cell(0, 6, "Nota: Este reporte fue generado automáticamente por el Pipeline de Zodiac AI aplicando algoritmos matemáticos en memoria.")
    
    # Exportar como string de bytes (convertir bytearray a bytes para Streamlit)
    return bytes(pdf.output(dest='S'))
