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
        
        # Inicializar respuestas Likert (p1 a p15) y abiertas (p1a a p15a) como None
        likert_resp = {f"p{p}": None for p in range(1, 16)}
        open_resp = {f"p{p}a": None for p in range(1, 16)}
        
        # Definimos las preguntas activas por elemento
        # Fuego: p1-p3
        # Agua: p4-p6
        # Aire: p7-p9
        # Tierra: p10-p12
        if elemento == "Fuego":
            preguntas_activas = [1, 2, 3]
        elif elemento == "Agua":
            preguntas_activas = [4, 5, 6]
        elif elemento == "Aire":
            preguntas_activas = [7, 8, 9]
        elif elemento == "Tierra":
            preguntas_activas = [10, 11, 12]
        else:
            preguntas_activas = []
            
        # Rellenar solo las preguntas activas correspondientes al elemento
        for p in preguntas_activas:
            # Sesgo alto (4 o 5) para simular fuerte identificación con su elemento
            val = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
            likert_resp[f"p{p}"] = val
            
            # Seleccionar frase semántica de texto libre
            frase = np.random.choice(FRASES_ELEMENTO[elemento])
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
    print("¡Dataset de 5,000 registros con estructura ramificada generado en 'data/datos_prueba_zodiac.csv'!")
