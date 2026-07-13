import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Diccionario simple de sentimientos en español para análisis local libre de descargas de NLTK/TextBlob en entornos offline
PALABRAS_POSITIVAS = {
    "apasionado", "creativo", "estable", "lider", "lealtad", "ordenado", "curioso", "adaptable", "empatica",
    "profunda", "tranquilidad", "entusiasmo", "honesto", "directo", "libertad", "iniciativa", "inteligente",
    "curiosidad", "alegre", "bueno", "practico", "realista", "logica", "sensible", "intuitivo", "ayudar"
}

PALABRAS_NEGATIVAS = {
    "desespero", "impaciente", "aburro", "rutina", "cambios", "atado", "dogmas", "estrictas", "introvertido",
    "nostalgico", "malo", "triste", "enojo", "miedo", "duda", "conflictivo", "pesado", "aburrido", "limite"
}

def analizar_sentimiento_local(texto):
    """
    Analiza el sentimiento de un texto en español de manera léxica simple.
    Retorna un valor de polaridad entre -1.0 (muy negativo) y 1.0 (muy positivo).
    """
    if not isinstance(texto, str) or not texto.strip():
        return 0.0
    
    palabras = re.findall(r'\b\w+\b', texto.lower())
    if not palabras:
        return 0.0
        
    pos_count = sum(1 for p in palabras if p in PALABRAS_POSITIVAS)
    neg_count = sum(1 for p in palabras if p in PALABRAS_NEGATIVAS)
    
    total = pos_count + neg_count
    if total == 0:
        # Intentar buscar subcadenas si no hay coincidencia exacta de palabras
        for p in palabras:
            for pos in PALABRAS_POSITIVAS:
                if pos in p:
                    pos_count += 0.5
            for neg in PALABRAS_NEGATIVAS:
                if neg in p:
                    neg_count += 0.5
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
    return (pos_count - neg_count) / total

def analizar_sentimientos_dataset(df, columnas_abiertas):
    """
    Calcula el sentimiento promedio para cada fila sumando el análisis de las columnas abiertas seleccionadas.
    """
    df_sentimientos = pd.DataFrame(index=df.index)
    
    for col in columnas_abiertas:
        if col in df.columns:
            df_sentimientos[f"{col}_polaridad"] = df[col].apply(analizar_sentimiento_local)
            
    # Sentimiento promedio global del sujeto
    df_sentimientos["sentimiento_promedio"] = df_sentimientos.mean(axis=1)
    return df_sentimientos

def obtener_terminos_frecuentes_tfidf(df, columnas_abiertas, max_features=10):
    """
    Calcula el TF-IDF global para las respuestas abiertas y extrae los términos más importantes.
    """
    textos = []
    for col in columnas_abiertas:
        if col in df.columns:
            textos.extend(df[col].dropna().astype(str).tolist())
            
    if not textos or len(textos) == 0 or all(t.strip() == "" for t in textos):
        return pd.DataFrame(columns=["Termino", "TF-IDF"])
        
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features) # 'english' por defecto, para español podemos pasar stop words básicas
        # Stop words en español básicas
        stop_words_es = ["el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no", "en", "para", "de", "con", "por", "que", "es", "me", "mi", "se", "muy", "como", "esta", "sobre", "pregunta"]
        vectorizer = TfidfVectorizer(stop_words=stop_words_es, max_features=max_features)
        
        tfidf_matrix = vectorizer.fit_transform(textos)
        feature_names = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0).A1
        
        df_tfidf = pd.DataFrame(list(zip(feature_names, sums)), columns=["Termino", "TF-IDF"])
        df_tfidf = df_tfidf.sort_values(by="TF-IDF", ascending=False).reset_index(drop=True)
        return df_tfidf
    except Exception as e:
        # Fallback simple por si falla la tokenización
        return pd.DataFrame({"Termino": ["error_procesamiento"], "TF-IDF": [0.0]})
