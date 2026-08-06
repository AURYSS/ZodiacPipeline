# 📊 Evidencia de Generación del Dataset Masivo (5,000 Registros)

Para entrenar y evaluar con éxito los modelos de clustering y las funciones NLP sin depender de encuestas reales sesgadas o incompletas, se diseñó un **Script Generador de Datos Sintéticos** en Python. 

Este script genera **5,000 registros coherentes**, estructurando datos demográficos, respuestas numéricas tipo Likert (`p1`-`p15`) y texto libre en español (`p1a`-`p15a`).

---

## 🧠 1. Lógica del Algoritmo de Generación

Para que los algoritmos de Clustering (K-Means, GMM, DBSCAN) tengan algo que aprender, los datos no pueden ser completamente aleatorios (ruido blanco). Por ello, el generador implementa una **lógica de sesgo temático por Elemento Zodiacal**:

### A. Segmentación de Signos por Elemento:
*   🔥 **Fuego** (Aries, Leo, Sagitario)
*   ⛰️ **Tierra** (Tauro, Virgo, Capricornio)
*   🌬️ **Aire** (Géminis, Libra, Acuario)
*   💧 **Agua** (Cáncer, Escorpio, Piscis)

### B. Distribución Orientada de Preguntas Likert (Escala 1 a 5):
El script aplica un sesgo probabilístico en las respuestas numéricas para simular patrones reales de personalidad astrológica:
*   Si el sujeto es de **Fuego**: Sus respuestas a las preguntas de liderazgo e impulso (`p1`, `p2`, `p3`) se sesgan hacia valores altos ($4$ y $5$).
*   Si el sujeto es de **Tierra**: Sus respuestas a preguntas de orden y estabilidad (`p4`, `p5`, `p6`) se sesgan hacia valores altos ($4$ y $5$).
*   Si el sujeto es de **Aire**: Se sesgan sus respuestas de sociabilidad y mente (`p7`, `p8`, `p9`).
*   Si el sujeto es de **Agua**: Se sesgan sus respuestas de sensibilidad y emoción (`p10`, `p11`, `p12`).

### C. Generación Semántica de Texto Libre (Respuestas Abiertas):
Se definieron diccionarios de frases específicas en español que representan el comportamiento lingüístico de cada elemento. Al generar el texto para `p1a` a `p15a`, el script selecciona aleatoriamente de estas plantillas según el signo del individuo. Esto proporciona las bases para que el módulo de **Análisis de Sentimiento** y **TF-IDF** encuentre palabras clave significativas (ej. *"apasionado"*, *"ordenado"*, *"curioso"*, *"nostálgico"*).

---

## 💻 2. Código del Script Generador (`generate_massive_data.py`)

A continuación se muestra el código completo en Python utilizado para fabricar la base de datos de prueba (`data/datos_prueba_zodiac.csv`):

```python
import pandas as pd
import numpy as np
import os

# Definición de Elementos y Signos
ELEMENTOS = {
    "Fuego": ["Aries", "Leo", "Sagitario"],
    "Tierra": ["Tauro", "Virgo", "Capricornio"],
    "Aire": ["Géminis", "Libra", "Acuario"],
    "Agua": ["Cáncer", "Escorpio", "Piscis"]
}

SIGNOS = []
SIGNOS_ELEMENTOS = {}
for elem, signs in ELEMENTOS.items():
    SIGNOS.extend(signs)
    for s in signs:
        SIGNOS_ELEMENTOS[s] = elem

# Plantillas de texto para respuestas abiertas basadas en la naturaleza de cada elemento
FRASES_ELEMENTO = {
    "Fuego": [
        "Soy impaciente pero honesto y directo con la gente, no me guardo lo que pienso.",
        "Tengo un temperamento fuerte. Me gusta liderar y suelo ser competitivo en lo que hago.",
        "Siento que tengo mucha energía, a veces me desespero rápido pero siempre sigo adelante con entusiasmo.",
        "Me considero una persona muy apasionada, impulsiva y me gusta tomar la iniciativa en proyectos.",
        "Busco siempre aventuras nuevas y me aburro fácilmente de la rutina diaria."
    ],
    "Tierra": [
        "Soy muy práctico y realista. Me esfuerzo por construir una base sólida para mi futuro.",
        "Valoro mucho la lealtad y el trabajo constante. No me gustan los cambios bruscos inesperados.",
        "Me gusta estar en contacto con la naturaleza y disfruto de la tranquilidad de mi hogar.",
        "Suelo ser reservado, analítico y busco siempre la lógica en las decisiones cotidianas.",
        "Prefiero la estabilidad, ser ordenado y planificar con tiempo mis actividades financieras."
    ],
    "Aire": [
        "Me adapto fácil a entornos sociales diversos y me gusta debatir sobre filosofía o ciencia.",
        "Disfruto mucho de conversar, intercambiar ideas complejas y conocer gente con diferentes puntos de vista.",
        "A veces divago mucho en mis pensamientos, pero se me ocurren ideas muy creativas y originales.",
        "Soy muy curioso, leo bastante y me gusta estar enterado de la tecnología y tendencias.",
        "Valoro la libertad intelectual, no me gusta sentirme atado a dogmas o reglas estrictas."
    ],
    "Agua": [
        "Soy una persona muy sensible, empática y guío mis decisiones a través de la intuición.",
        "Siento las emociones de forma muy intensa y busco conexiones profundas con mis seres queridos.",
        "A menudo necesito tiempo a solas para recargar mi energía emocional y reflexionar.",
        "Me conmuevo fácil con el arte y la música. A veces tiendo a ser introvertido y nostálgico.",
        "Me gusta ayudar a los demás y soy bastante perceptivo con los estados de ánimo ajenos."
    ]
}

def generar_dataset(n_registros=5000):
    np.random.seed(42)  # Garantizar reproducibilidad
    
    data = []
    
    for i in range(1, n_registros + 1):
        # Datos demográficos
        edad = int(np.random.randint(18, 70))
        genero = np.random.choice(["Femenino", "Masculino", "No binario"], p=[0.48, 0.48, 0.04])
        signo = np.random.choice(SIGNOS)
        elemento = SIGNOS_ELEMENTOS[signo]
        
        # Respuestas Likert (1 a 5)
        likert_resp = {}
        
        # Definimos sesgos de respuesta según el elemento astrológico
        for p in range(1, 16):
            if elemento == "Fuego" and p in [1, 2, 3]:
                # Sesgo alto para Fuego en preguntas 1, 2, 3
                val = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
            elif elemento == "Tierra" and p in [4, 5, 6]:
                # Sesgo alto para Tierra en preguntas 4, 5, 6
                val = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
            elif elemento == "Aire" and p in [7, 8, 9]:
                # Sesgo alto para Aire en preguntas 7, 8, 9
                val = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
            elif elemento == "Agua" and p in [10, 11, 12]:
                # Sesgo alto para Agua en preguntas 10, 11, 12
                val = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
            else:
                # Distribución uniforme normal para las demás preguntas
                val = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.20, 0.30, 0.20, 0.15])
            likert_resp[f"p{p}"] = val
            
        # Respuestas abiertas (p1a a p15a)
        open_resp = {}
        for p in range(1, 16):
            # Elige una frase aleatoria de la lista correspondiente a su elemento
            frase = np.random.choice(FRASES_ELEMENTO[elemento])
            # Formatear la respuesta abierta
            open_resp[f"p{p}a"] = f"Pregunta {p}: {frase}"
            
        # Unir todas las columnas del registro
        registro = {
            "id": i,
            "edad": edad,
            "genero": genero,
            "signo": signo
        }
        registro.update(likert_resp)
        registro.update(open_resp)
        data.append(registro)
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df_prueba = generar_dataset(5000)
    # Crear directorio si no existe y exportar
    os.makedirs("data", exist_ok=True)
    df_prueba.to_csv("data/datos_prueba_zodiac.csv", index=False)
    print("¡Dataset de 5,000 registros generado exitosamente en 'data/datos_prueba_zodiac.csv'!")
```

---

## 📈 3. ¿Por qué es evidencia válida?
1.  **Uniformidad Estructural**: Sigue exactamente el formato de base de datos requerido por la tabla `encuestas` de PostgreSQL.
2.  **Facilita el Clustering**: Al sesgar los grupos de preguntas Likert por elementos astrológicos, garantiza que algoritmos como K-Means y GMM puedan encontrar patrones y separar grupos de forma matemática clara (alto *Silhouette Score*).
3.  **Procesamiento de Lenguaje Natural**: Las respuestas abiertas insertadas contienen semántica orientada (positivo/negativo y términos repetidos), permitiendo que la extracción TF-IDF identifique los descriptores correctos para cada perfil.
