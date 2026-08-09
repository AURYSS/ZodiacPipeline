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

def calcular_mediana_manual(valores):
    """
    Calcula la mediana (Me) de forma manual.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    if not limpios:
        return None
    n = len(limpios)
    # Ordenamiento manual simple (burbuja) para fines académicos o simplemente sort()
    # Usaremos sort nativo por eficiencia, pero el cálculo posicional es manual.
    limpios.sort()
    
    if n % 2 == 0:
        mitad1 = limpios[n//2 - 1]
        mitad2 = limpios[n//2]
        return (float(mitad1) + float(mitad2)) / 2.0
    else:
        return float(limpios[n//2])

def calcular_cv_manual(valores):
    """
    Calcula el Coeficiente de Variación (CV).
    Fórmula: (Desviación Estándar / Media) * 100
    """
    media = calcular_promedio_manual(valores)
    if media is None or media == 0:
        return 0.0
    std = calcular_desviacion_estandar_manual(valores)
    return (std / float(media)) * 100.0


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

def calcular_minimo_manual(valores):
    """
    Calcula el mínimo de una lista de números.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    if not limpios:
        return None
    val_min = limpios[0]
    for x in limpios:
        if x < val_min:
            val_min = x
    return val_min

def calcular_maximo_manual(valores):
    """
    Calcula el máximo de una lista de números.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    if not limpios:
        return None
    val_max = limpios[0]
    for x in limpios:
        if x > val_max:
            val_max = x
    return val_max

def calcular_rango_manual(valores):
    """
    Calcula el rango (máximo - mínimo).
    """
    val_min = calcular_minimo_manual(valores)
    val_max = calcular_maximo_manual(valores)
    if val_min is None or val_max is None:
        return 0.0
    return float(val_max) - float(val_min)

def calcular_k_sturges(n):
    """
    Calcula el número de clases (K) usando la Regla de Sturges.
    K = 1 + 3.322 * log10(n)
    """
    if n <= 0:
        return 0
    return 1 + 3.322 * np.log10(n)

def calcular_amplitud_manual(rango, k):
    """
    Calcula la amplitud (Rango / K).
    """
    if k <= 0:
        return 0.0
    return float(rango) / float(k)

def generar_tabla_frecuencias_manual(valores):
    """
    Genera la tabla de frecuencias matemática (Marca de Clase, f, Fr, %, F).
    Retorna un DataFrame con la tabla construida paso a paso.
    """
    limpios = [x for x in valores if pd.notnull(x)]
    n = len(limpios)
    if n == 0:
        return pd.DataFrame()
        
    val_min = calcular_minimo_manual(limpios)
    val_max = calcular_maximo_manual(limpios)
    rango = float(val_max) - float(val_min)
    k_float = calcular_k_sturges(n)
    # Según apuntes escolares, K se suele redondear al entero más cercano
    k_int = int(round(k_float))
    if k_int < 1:
        k_int = 1
        
    amplitud = calcular_amplitud_manual(rango, k_int)
    
    # Construcción de intervalos
    tabla = []
    limite_inf = float(val_min)
    frecuencia_acumulada = 0
    
    for i in range(k_int):
        limite_sup = limite_inf + amplitud
        
        # Conteo de frecuencias (f) manual
        frecuencia_absoluta = 0
        for val in limpios:
            # En el último intervalo cerramos corchete por la derecha
            if i == k_int - 1:
                if limite_inf <= val <= limite_sup:
                    frecuencia_absoluta += 1
            else:
                if limite_inf <= val < limite_sup:
                    frecuencia_absoluta += 1
                    
        # Cálculos de clase
        marca_clase = (limite_inf + limite_sup) / 2.0
        frecuencia_relativa = frecuencia_absoluta / float(n) if n > 0 else 0
        porcentaje = frecuencia_relativa * 100.0
        frecuencia_acumulada += frecuencia_absoluta
        
        # Etiqueta del intervalo
        if i == k_int - 1:
            rango_str = f"[{round(limite_inf, 2)}, {round(limite_sup, 2)}]"
        else:
            rango_str = f"[{round(limite_inf, 2)}, {round(limite_sup, 2)})"
            
        tabla.append({
            "Clase": rango_str,
            "Marca de Clase (x)": round(marca_clase, 4),
            "f": frecuencia_absoluta,
            "Fr": round(frecuencia_relativa, 4),
            "%": round(porcentaje, 2),
            "F": frecuencia_acumulada
        })
        
        limite_inf = limite_sup
        
    # Fila de Totales
    tabla.append({
        "Clase": "Total",
        "Marca de Clase (x)": "-",
        "f": n,
        "Fr": 1.0000,
        "%": 100.00,
        "F": "-"
    })
    
    return pd.DataFrame(tabla)
