import pandas as pd

def load_and_validate_csv(file_path_or_buffer):
    """
    Carga un archivo CSV y valida que contenga la estructura requerida:
    id, edad, genero, signo, p1..p15, p1a..p15a
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        return None, f"Error al leer el archivo CSV: {str(e)}"
    
    # Validaciones de columnas requeridas
    required_cols = ["id", "edad", "genero", "signo"]
    likert_cols = [f"p{i}" for i in range(1, 16)]
    open_cols = [f"p{i}a" for i in range(1, 16)]
    
    missing_required = [col for col in required_cols if col not in df.columns]
    missing_likert = [col for col in likert_cols if col not in df.columns]
    missing_open = [col for col in open_cols if col not in df.columns]
    
    errors = []
    if missing_required:
        errors.append(f"Faltan columnas de identificación/demografía obligatorias: {', '.join(missing_required)}")
    if missing_likert:
        errors.append(f"Faltan preguntas Likert obligatorias (p1 a p15): {', '.join(missing_likert)}")
    if missing_open:
        errors.append(f"Faltan preguntas abiertas obligatorias (p1a a p15a): {', '.join(missing_open)}")
        
    if errors:
        return None, " | ".join(errors)
        
    return df, "Validación exitosa"
