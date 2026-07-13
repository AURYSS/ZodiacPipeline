import pandas as pd
import numpy as np

def calcular_promedio_manual(valores):
    """
    Calcula el promedio manual de una lista/serie de números.
    Rúbrica: Algoritmo propio.
    """
    # Filtramos nulos
    limpios = [x for x in valores if pd.notnull(x)]
    if not limpios:
        return 0.0
    suma = 0.0
    for x in limpios:
        suma += float(x)
    return suma / len(limpios)

def calcular_moda_manual(valores):
    """
    Calcula la moda manual (el valor más común) de una lista/serie.
    Rúbrica: Algoritmo propio.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    if not limpios:
        return None
    frecuencias = {}
    for x in limpios:
        frecuencias[x] = frecuencias.get(x, 0) + 1
    
    max_freq = -1
    moda = None
    for val, freq in frecuencias.items():
        if freq > max_freq:
            max_freq = freq
            moda = val
    return moda

def calcular_varianza_manual(valores):
    """
    Calcula la varianza muestral manual (con divisor N-1).
    Rúbrica: Algoritmo propio.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    n = len(limpios)
    if n < 2:
        return 0.0
    promedio = calcular_promedio_manual(limpios)
    suma_cuadrados = 0.0
    for x in limpios:
        suma_cuadrados += (float(x) - promedio) ** 2
    return suma_cuadrados / (n - 1)

def calcular_desviacion_estandar_manual(valores):
    """
    Calcula la desviación estándar manual.
    Rúbrica: Algoritmo propio.
    """
    var = calcular_varianza_manual(valores)
    return var ** 0.5

def normalizacion_min_max_manual(df, columnas):
    """
    Normaliza de manera manual las columnas seleccionadas del dataframe al rango [0, 1].
    Fórmula: (x - min) / (max - min)
    Rúbrica: Algoritmo propio.
    """
    df_norm = df.copy()
    for col in columnas:
        # Extraer valores numéricos limpios para hallar min y max manualmente
        valores = df[col].dropna().tolist()
        if not valores:
            continue
        
        # Algoritmo de min y max manual
        val_min = valores[0]
        val_max = valores[0]
        for val in valores:
            if val < val_min:
                val_min = val
            if val > val_max:
                val_max = val
        
        rango = val_max - val_min
        if rango == 0:
            df_norm[col] = 0.0
        else:
            # Reemplazar manualmente
            df_norm[col] = df[col].apply(lambda x: (x - val_min) / rango if pd.notnull(x) else 0.0)
            
    return df_norm

def calcular_promedios_dimensiones(df):
    """
    Calcula promedios por dimensión según la agrupación temática:
    Fuego: P1 - P3
    Tierra: P4 - P6
    Aire: P7 - P9
    Agua: P10 - P12
    General/Polaridad: P13 - P15
    """
    dimensiones = {
        "Fuego (Impulso/Liderazgo)": ["p1", "p2", "p3"],
        "Tierra (Estabilidad/Orden)": ["p4", "p5", "p6"],
        "Aire (Sociabilidad/Mente)": ["p7", "p8", "p9"],
        "Agua (Sensibilidad/Emoción)": ["p10", "p11", "p12"],
        "General/Polaridades": ["p13", "p14", "p15"]
    }
    
    resultados = {}
    for dim_name, cols in dimensiones.items():
        # Validamos que existan
        cols_presentes = [c for c in cols if c in df.columns]
        if cols_presentes:
            # Obtenemos la serie promedio para cada fila usando nuestra función manual
            resultados[dim_name] = df[cols_presentes].apply(lambda row: calcular_promedio_manual(row), axis=1)
            
    return pd.DataFrame(resultados)
