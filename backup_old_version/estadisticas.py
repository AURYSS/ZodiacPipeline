"""
estadisticas.py
----------------
Funciones de estadística descriptiva implementadas "a mano" (algoritmos
propios), sin depender de pandas.describe() ni de numpy para el cálculo
en sí. Se usan listas y operaciones básicas de Python, tal como pide la
actividad: "Generación de información estadística básica, con base a
algoritmos propios".
"""

from collections import Counter


def calcular_media(valores):
    """Promedio aritmético calculado manualmente."""
    if not valores:
        return None
    return sum(valores) / len(valores)


def calcular_mediana(valores):
    """Valor central de la lista ordenada."""
    if not valores:
        return None
    ordenados = sorted(valores)
    n = len(ordenados)
    mitad = n // 2
    if n % 2 == 0:
        return (ordenados[mitad - 1] + ordenados[mitad]) / 2
    return ordenados[mitad]


def calcular_moda(valores):
    """Valor(es) más frecuente(s)."""
    if not valores:
        return None
    conteo = Counter(valores)
    frecuencia_max = max(conteo.values())
    modas = [valor for valor, freq in conteo.items() if freq == frecuencia_max]
    # Si todos los valores tienen la misma frecuencia, no hay moda representativa
    if len(modas) == len(conteo):
        return None
    return sorted(modas)


def calcular_varianza(valores):
    """Varianza poblacional, calculada a partir de la media manual."""
    if not valores or len(valores) < 2:
        return 0.0
    media = calcular_media(valores)
    suma_cuadrados = sum((x - media) ** 2 for x in valores)
    return suma_cuadrados / len(valores)


def calcular_desviacion_estandar(valores):
    """Raíz cuadrada de la varianza (implementación manual, sin math.sqrt** para exponente)."""
    varianza = calcular_varianza(valores)
    return varianza ** 0.5


def calcular_rango(valores):
    """Diferencia entre el valor máximo y el mínimo."""
    if not valores:
        return None
    return max(valores) - min(valores)


def calcular_cuartiles(valores):
    """Q1, Q2 (mediana) y Q3 usando el método de posición sencillo."""
    if not valores:
        return None
    ordenados = sorted(valores)
    n = len(ordenados)

    def _posicion(p):
        indice = p * (n - 1)
        piso = int(indice)
        resto = indice - piso
        if piso + 1 < n:
            return ordenados[piso] + resto * (ordenados[piso + 1] - ordenados[piso])
        return ordenados[piso]

    return {
        "Q1": round(_posicion(0.25), 2),
        "Q2": round(_posicion(0.50), 2),
        "Q3": round(_posicion(0.75), 2),
    }


def generar_resumen_estadistico(registros, columnas_numericas):
    """
    Genera un resumen estadístico básico para cada columna numérica.

    registros: lista de diccionarios (cada uno es una fila del dataset)
    columnas_numericas: lista de nombres de columnas a analizar
    """
    resumen = {}
    for columna in columnas_numericas:
        valores = []
        for fila in registros:
            valor = fila.get(columna)
            if valor is None or valor == "":
                continue
            try:
                valores.append(float(valor))
            except (TypeError, ValueError):
                continue

        if not valores:
            continue

        resumen[columna] = {
            "n": len(valores),
            "media": round(calcular_media(valores), 2),
            "mediana": round(calcular_mediana(valores), 2),
            "moda": calcular_moda(valores),
            "varianza": round(calcular_varianza(valores), 2),
            "desviacion_estandar": round(calcular_desviacion_estandar(valores), 2),
            "minimo": min(valores),
            "maximo": max(valores),
            "rango": calcular_rango(valores),
            "cuartiles": calcular_cuartiles(valores),
        }
    return resumen


def contar_frecuencias_categoricas(registros, columna_categorica):
    """Distribución de frecuencias de una columna categórica (p. ej. 'elemento')."""
    conteo = Counter(fila.get(columna_categorica) for fila in registros if fila.get(columna_categorica))
    total = sum(conteo.values())
    return {
        valor: {"conteo": cantidad, "porcentaje": round((cantidad / total) * 100, 1)}
        for valor, cantidad in sorted(conteo.items())
    }
