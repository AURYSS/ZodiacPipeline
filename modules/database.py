import os
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

# Parámetros de conexión a la base de datos PostgreSQL local
DB_NAME = os.getenv("DB_NAME", "zodiac")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Aur0ra")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_connection():
    """
    Establece y retorna una conexión a la base de datos local PostgreSQL.
    """
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def init_db():
    """
    Inicializa la base de datos creando la tabla 'encuestas' si no existe.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Crear la tabla 'encuestas'
    # Definir campos numéricos y abiertos
    likert_fields = ", ".join([f"p{i} INT" for i in range(1, 16)])
    open_fields = ", ".join([f"p{i}a TEXT" for i in range(1, 16)])
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS encuestas (
        id SERIAL PRIMARY KEY,
        edad INT,
        genero VARCHAR(50),
        signo VARCHAR(50),
        {likert_fields},
        {open_fields}
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()
    print("Base de datos y tabla 'encuestas' inicializadas con éxito en Postgres.")

def insertar_encuestas(df):
    """
    Inserta un lote de encuestas (DataFrame) a la tabla 'encuestas'.
    """
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    
    # Asegurar orden exacto de columnas para la inserción
    cols_likert = [f"p{i}" for i in range(1, 16)]
    cols_open = [f"p{i}a" for i in range(1, 16)]
    columnas_ordenadas = ["edad", "genero", "signo"] + cols_likert + cols_open
    
    # Preparar datos
    datos_insercion = df[columnas_ordenadas].values.tolist()
    
    # Consulta de inserción masiva
    cols_str = ", ".join(columnas_ordenadas)
    query = f"INSERT INTO encuestas ({cols_str}) VALUES %s"
    
    execute_values(cur, query, datos_insercion)
    conn.commit()
    
    inserted_count = len(datos_insercion)
    cur.close()
    conn.close()
    return inserted_count

def obtener_todas_las_encuestas():
    """
    Recupera todas las encuestas guardadas en la base de datos PostgreSQL.
    Retorna un Pandas DataFrame.
    """
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM encuestas ORDER BY id ASC;"
    cur.execute(query)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=colnames)

def limpiar_base_de_datos():
    """
    Limpia todos los registros de la tabla 'encuestas'.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE encuestas RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()
