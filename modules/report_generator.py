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
        
    if graphs_bytes:
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.set_text_color(74, 14, 78)
        pdf.cell(0, 10, '4. Visualizaciones Analíticas', ln=True)
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
