import pandas as pd
import numpy as np
import re

def parse_likert_value(val):
    """
    Parsea una respuesta Likert de Google Forms (ej. '5 = Totalmente de acuerdo')
    y retorna el valor entero correspondiente (5). Si es nulo, retorna None.
    """
    if pd.isnull(val):
        return None
    val_str = str(val).strip()
    if not val_str:
        return None
    # Intentar buscar un dígito del 1 al 5 al inicio
    match = re.match(r'^([1-5])', val_str)
    if match:
        return int(match.group(1))
    
    # Si no coincide, intentar convertir a entero directamente
    try:
        return int(float(val_str))
    except ValueError:
        return None

def parse_age(val):
    """
    Mapea los rangos de edad del formulario de Google a un entero representativo (punto medio).
    """
    if pd.isnull(val):
        return 30 # Valor por defecto
    val_str = str(val).strip()
    
    # Mapeo de rangos estándar del Google Form
    if "18 - 25" in val_str:
        return 21
    elif "26 - 35" in val_str:
        return 30
    elif "36 - 45" in val_str:
        return 40
    elif "Mayor a 45" in val_str:
        return 52
    elif "Menor o igual a 17" in val_str:
        return 16
    
    # Si es numérico directo
    try:
        return int(float(val_str))
    except ValueError:
        return 30

def parse_element_to_sign(val):
    """
    Mapea la opción del elemento astrológico seleccionado al nombre del elemento.
    """
    if pd.isnull(val):
        return "Desconocido"
    val_str = str(val).lower()
    if "fuego" in val_str:
        return "Fuego"
    elif "tierra" in val_str:
        return "Tierra"
    elif "aire" in val_str:
        return "Aire"
    elif "agua" in val_str:
        return "Agua"
    return "Desconocido"

def map_google_form_df(df):
    """
    Mapea las columnas largas del reporte de Google Forms a la estructura de base de datos encuestas.
    """
    mapped_data = []
    
    for idx, row in df.iterrows():
        # Demografía
        # Col 2: Género, Col 3: Edad, Col 4: Elemento
        genero = row.iloc[2] if len(row) > 2 else "Otro"
        edad = parse_age(row.iloc[3]) if len(row) > 3 else 30
        signo = parse_element_to_sign(row.iloc[4]) if len(row) > 4 else "Desconocido"
        
        # Mapeo de preguntas de 1 a 12 según la columna correspondiente de Google Form
        # Fuego: p1-p3 (indices de columnas 5, 7, 9) y textos (indices 6, 8, 10)
        # Agua: p4-p6 (indices de columnas 11, 13, 15) y textos (indices 12, 14, 16)
        # Aire: p7-p9 (indices de columnas 17, 19, 21) y textos (indices 18, 20, 22)
        # Tierra: p10-p12 (indices de columnas 23, 25, 27) y textos (indices 24, 26, 28)
        
        # Inicializar todo como None
        likert_vals = {f"p{p}": None for p in range(1, 16)}
        open_texts = {f"p{p}a": None for p in range(1, 16)}
        
        # Fuego
        if len(row) > 5: likert_vals["p1"] = parse_likert_value(row.iloc[5])
        if len(row) > 6: open_texts["p1a"] = str(row.iloc[6]) if pd.notnull(row.iloc[6]) and str(row.iloc[6]).strip() else None
        if len(row) > 7: likert_vals["p2"] = parse_likert_value(row.iloc[7])
        if len(row) > 8: open_texts["p2a"] = str(row.iloc[8]) if pd.notnull(row.iloc[8]) and str(row.iloc[8]).strip() else None
        if len(row) > 9: likert_vals["p3"] = parse_likert_value(row.iloc[9])
        if len(row) > 10: open_texts["p3a"] = str(row.iloc[10]) if pd.notnull(row.iloc[10]) and str(row.iloc[10]).strip() else None
        
        # Agua
        if len(row) > 11: likert_vals["p4"] = parse_likert_value(row.iloc[11])
        if len(row) > 12: open_texts["p4a"] = str(row.iloc[12]) if pd.notnull(row.iloc[12]) and str(row.iloc[12]).strip() else None
        if len(row) > 13: likert_vals["p5"] = parse_likert_value(row.iloc[13])
        if len(row) > 14: open_texts["p5a"] = str(row.iloc[14]) if pd.notnull(row.iloc[14]) and str(row.iloc[14]).strip() else None
        if len(row) > 15: likert_vals["p6"] = parse_likert_value(row.iloc[15])
        if len(row) > 16: open_texts["p6a"] = str(row.iloc[16]) if pd.notnull(row.iloc[16]) and str(row.iloc[16]).strip() else None
        
        # Aire
        if len(row) > 17: likert_vals["p7"] = parse_likert_value(row.iloc[17])
        if len(row) > 18: open_texts["p7a"] = str(row.iloc[18]) if pd.notnull(row.iloc[18]) and str(row.iloc[18]).strip() else None
        if len(row) > 19: likert_vals["p8"] = parse_likert_value(row.iloc[19])
        if len(row) > 20: open_texts["p8a"] = str(row.iloc[20]) if pd.notnull(row.iloc[20]) and str(row.iloc[20]).strip() else None
        if len(row) > 21: likert_vals["p9"] = parse_likert_value(row.iloc[21])
        if len(row) > 22: open_texts["p9a"] = str(row.iloc[22]) if pd.notnull(row.iloc[22]) and str(row.iloc[22]).strip() else None
        
        # Tierra
        if len(row) > 23: likert_vals["p10"] = parse_likert_value(row.iloc[23])
        if len(row) > 24: open_texts["p10a"] = str(row.iloc[24]) if pd.notnull(row.iloc[24]) and str(row.iloc[24]).strip() else None
        if len(row) > 25: likert_vals["p11"] = parse_likert_value(row.iloc[25])
        if len(row) > 26: open_texts["p11a"] = str(row.iloc[26]) if pd.notnull(row.iloc[26]) and str(row.iloc[26]).strip() else None
        if len(row) > 27: likert_vals["p12"] = parse_likert_value(row.iloc[27])
        if len(row) > 28: open_texts["p12a"] = str(row.iloc[28]) if pd.notnull(row.iloc[28]) and str(row.iloc[28]).strip() else None
        
        # Armar fila
        new_row = {
            "id": idx + 1,
            "edad": edad,
            "genero": genero,
            "signo": signo
        }
        new_row.update(likert_vals)
        new_row.update(open_texts)
        mapped_data.append(new_row)
        
    return pd.DataFrame(mapped_data)

def load_and_validate_csv(file_path_or_buffer):
    """
    Carga un archivo CSV o Excel y valida/transforma su estructura.
    Soporta tanto el formato nativo del pipeline (p1..p15) como el formato raw de Google Forms.
    """
    try:
        # Detectamos si es excel por el nombre del archivo
        filename = getattr(file_path_or_buffer, "name", "").lower()
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        return None, f"Error al leer el archivo: {str(e)}"
    
    # Comprobar si es formato Google Forms (ej. contiene 'Marca temporal' en columnas o tiene más de 25 columnas)
    is_google_form = any("marca temporal" in str(col).lower() for col in df.columns) or (len(df.columns) >= 28 and any("signo zodiacal" in str(col).lower() for col in df.columns))
    
    if is_google_form:
        print("Estructura Google Forms detectada. Mapeando columnas...")
        try:
            df_mapped = map_google_form_df(df)
            return df_mapped, "Google Forms mapeado exitosamente."
        except Exception as e:
            return None, f"Error al procesar el formato de Google Forms: {str(e)}"
            
    # Si es formato nativo estándar, realizar validaciones ordinarias
    required_cols = ["id", "edad", "genero", "signo"]
    likert_cols = [f"p{i}" for i in range(1, 16)]
    open_cols = [f"p{i}a" for i in range(1, 16)]
    
    missing_required = [col for col in required_cols if col not in df.columns]
    missing_likert = [col for col in likert_cols if col not in df.columns]
    missing_open = [col for col in open_cols if col not in df.columns]
    
    errors = []
    if missing_required:
        errors.append(f"Faltan columnas obligatorias: {', '.join(missing_required)}")
    if missing_likert:
        errors.append(f"Faltan preguntas Likert (p1 a p15): {', '.join(missing_likert)}")
    if missing_open:
        errors.append(f"Faltan preguntas abiertas (p1a a p15a): {', '.join(missing_open)}")
        
    if errors:
        return None, " | ".join(errors)
        
    # Limpiar columnas Likert de posibles textos si es necesario (ej. si vienen strings en vez de ints)
    for col in likert_cols:
        df[col] = df[col].apply(parse_likert_value)
        
    return df, "Validación exitosa"
