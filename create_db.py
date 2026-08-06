import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    conn = psycopg2.connect(dbname='postgres', user='postgres', password='Aur0ra', host='localhost', port='5432')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute('CREATE DATABASE zodiac;')
    print("Exito")
    cursor.close()
    conn.close()
except psycopg2.errors.DuplicateDatabase:
    print("La base de datos ya existe.")
except Exception as e:
    print(f"Error: {e}")
