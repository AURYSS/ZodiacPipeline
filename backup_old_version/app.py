"""
app.py
------
Aplicativo Flask para la recolección y análisis de perfiles de usuario
respecto a su signo zodiacal, aplicando algoritmos no supervisados
(K-Means y Clusterización Jerárquica).

Funcionalidades cubiertas (según la actividad):
  - Carga de datos (CSV)
  - Mostrar la información cargada
  - Filtro por categorías (elemento / signo zodiacal)
  - Estadística básica con algoritmos propios
  - Entrenamiento del algoritmo, generación y guardado del modelo
  - Generación de resultados tras el aprendizaje
  - Descarga de datos filtrados y de resultados
"""

import os
import csv
import io
from flask import Flask, request, jsonify, render_template, send_file, session

import estadisticas
import modelos

app = Flask(__name__)
app.secret_key = "cambia-esta-llave-en-produccion"

BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATA_ACTUAL = os.path.join(UPLOADS_DIR, "dataset_actual.csv")
os.makedirs(UPLOADS_DIR, exist_ok=True)

COLUMNAS_NUMERICAS_DEFAULT = ["edad", "energia", "creatividad", "sociabilidad",
                               "estabilidad_emocional", "ambicion"]
COLUMNA_CATEGORICA_DEFAULT = "elemento"

# Guardamos en memoria el último resultado de entrenamiento para poder
# generar la descarga de "resultados" sin volver a entrenar.
ULTIMO_RESULTADO = {"datos": None}


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------
def _leer_dataset_actual():
    """Lee el CSV actualmente cargado y lo devuelve como lista de dicts."""
    if not os.path.exists(DATA_ACTUAL):
        return []
    with open(DATA_ACTUAL, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _escribir_csv_en_memoria(registros, columnas):
    """Genera un CSV en memoria (BytesIO) a partir de una lista de dicts."""
    buffer_texto = io.StringIO()
    writer = csv.DictWriter(buffer_texto, fieldnames=columnas)
    writer.writeheader()
    for fila in registros:
        writer.writerow({col: fila.get(col, "") for col in columnas})

    buffer_bytes = io.BytesIO(buffer_texto.getvalue().encode("utf-8"))
    buffer_bytes.seek(0)
    return buffer_bytes


def _aplicar_filtro(registros, columna, valor):
    if not columna or not valor:
        return registros
    return [fila for fila in registros if fila.get(columna) == valor]


# ----------------------------------------------------------------------
# Vista principal
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# 1) Carga de datos
# ----------------------------------------------------------------------
@app.route("/api/cargar", methods=["POST"])
def cargar_datos():
    archivo = request.files.get("archivo")
    if archivo is None or archivo.filename == "":
        return jsonify({"error": "No se recibió ningún archivo CSV."}), 400

    if not archivo.filename.lower().endswith(".csv"):
        return jsonify({"error": "El archivo debe tener extensión .csv"}), 400

    archivo.save(DATA_ACTUAL)
    registros = _leer_dataset_actual()

    return jsonify({
        "mensaje": f"Archivo cargado correctamente ({len(registros)} registros).",
        "total_registros": len(registros),
        "columnas": list(registros[0].keys()) if registros else [],
    })


@app.route("/api/cargar-ejemplo", methods=["POST"])
def cargar_ejemplo():
    """Carga el dataset de ejemplo incluido en /data para pruebas rápidas."""
    ruta_ejemplo = os.path.join(BASE_DIR, "data", "perfil_zodiacal_ejemplo.csv")
    with open(ruta_ejemplo, "rb") as origen, open(DATA_ACTUAL, "wb") as destino:
        destino.write(origen.read())
    registros = _leer_dataset_actual()
    return jsonify({
        "mensaje": f"Dataset de ejemplo cargado ({len(registros)} registros).",
        "total_registros": len(registros),
        "columnas": list(registros[0].keys()) if registros else [],
    })


# ----------------------------------------------------------------------
# 2) Mostrar información cargada  +  3) Filtro por categorías
# ----------------------------------------------------------------------
@app.route("/api/datos", methods=["GET"])
def obtener_datos():
    registros = _leer_dataset_actual()
    columna_filtro = request.args.get("columna")
    valor_filtro = request.args.get("valor")
    registros = _aplicar_filtro(registros, columna_filtro, valor_filtro)

    return jsonify({
        "total": len(registros),
        "registros": registros,
    })


@app.route("/api/categorias", methods=["GET"])
def obtener_categorias():
    """Devuelve los valores únicos disponibles para filtrar (por columna)."""
    registros = _leer_dataset_actual()
    columna = request.args.get("columna", COLUMNA_CATEGORICA_DEFAULT)
    valores = sorted({fila.get(columna) for fila in registros if fila.get(columna)})
    return jsonify({"columna": columna, "valores": valores})


# ----------------------------------------------------------------------
# 4) Estadística básica (algoritmos propios)
# ----------------------------------------------------------------------
@app.route("/api/estadisticas", methods=["GET"])
def obtener_estadisticas():
    registros = _leer_dataset_actual()
    columna_filtro = request.args.get("columna")
    valor_filtro = request.args.get("valor")
    registros = _aplicar_filtro(registros, columna_filtro, valor_filtro)

    if not registros:
        return jsonify({"error": "No hay datos cargados."}), 400

    resumen_numerico = estadisticas.generar_resumen_estadistico(registros, COLUMNAS_NUMERICAS_DEFAULT)
    frecuencias_elemento = estadisticas.contar_frecuencias_categoricas(registros, "elemento")
    frecuencias_signo = estadisticas.contar_frecuencias_categoricas(registros, "signo_zodiacal")

    return jsonify({
        "total_registros": len(registros),
        "resumen_numerico": resumen_numerico,
        "frecuencias_elemento": frecuencias_elemento,
        "frecuencias_signo": frecuencias_signo,
    })


# ----------------------------------------------------------------------
# 5) Entrenamiento del modelo  +  6) Resultados del aprendizaje
# ----------------------------------------------------------------------
@app.route("/api/entrenar", methods=["POST"])
def entrenar():
    cuerpo = request.get_json(force=True)
    algoritmo = cuerpo.get("algoritmo", "kmeans")
    k = int(cuerpo.get("k", 4))
    columnas = cuerpo.get("columnas") or COLUMNAS_NUMERICAS_DEFAULT
    nombre_modelo = cuerpo.get("nombre_modelo", f"modelo_{algoritmo}")

    registros = _leer_dataset_actual()
    columna_filtro = cuerpo.get("columna_filtro")
    valor_filtro = cuerpo.get("valor_filtro")
    registros = _aplicar_filtro(registros, columna_filtro, valor_filtro)

    if not registros:
        return jsonify({"error": "No hay datos cargados para entrenar."}), 400

    try:
        if algoritmo == "kmeans":
            modelo, scaler, resultado = modelos.entrenar_kmeans(registros, columnas, k=k)
        elif algoritmo == "jerarquico":
            metodo_enlace = cuerpo.get("metodo_enlace", "ward")
            modelo, scaler, resultado = modelos.entrenar_jerarquico(registros, columnas, k=k, metodo_enlace=metodo_enlace)
        else:
            return jsonify({"error": "Algoritmo no soportado. Usa 'kmeans' o 'jerarquico'."}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Guardar el modelo entrenado para uso posterior
    ruta_guardado = modelos.guardar_modelo(modelo, scaler, columnas, nombre_modelo)

    # Anexar la etiqueta de clúster a cada registro para mostrar resultados
    registros_con_resultado = []
    for fila, etiqueta in zip(registros, resultado["etiquetas"]):
        nueva_fila = dict(fila)
        nueva_fila["cluster"] = etiqueta
        registros_con_resultado.append(nueva_fila)

    ULTIMO_RESULTADO["datos"] = {
        "registros": registros_con_resultado,
        "columnas": list(registros_con_resultado[0].keys()) if registros_con_resultado else [],
    }

    resultado["nombre_modelo"] = nombre_modelo
    resultado["ruta_modelo"] = ruta_guardado
    resultado["registros_resultado"] = registros_con_resultado

    return jsonify(resultado)


@app.route("/api/modelos", methods=["GET"])
def listar_modelos():
    return jsonify({"modelos": modelos.listar_modelos_guardados()})


@app.route("/api/clasificar", methods=["POST"])
def clasificar_usuario():
    """Clasifica un nuevo perfil de usuario usando un modelo K-Means ya guardado."""
    cuerpo = request.get_json(force=True)
    nombre_modelo = cuerpo.get("nombre_modelo")
    valores = cuerpo.get("valores")  # dict con las columnas numéricas

    try:
        cluster = modelos.clasificar_nuevo_registro(nombre_modelo, valores)
    except (FileNotFoundError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"cluster_asignado": cluster})


# ----------------------------------------------------------------------
# 7) Descarga de datos filtrados y de resultados
# ----------------------------------------------------------------------
@app.route("/api/descargar/datos", methods=["GET"])
def descargar_datos_filtrados():
    registros = _leer_dataset_actual()
    columna_filtro = request.args.get("columna")
    valor_filtro = request.args.get("valor")
    registros = _aplicar_filtro(registros, columna_filtro, valor_filtro)

    if not registros:
        return jsonify({"error": "No hay datos para descargar."}), 400

    columnas = list(registros[0].keys())
    buffer = _escribir_csv_en_memoria(registros, columnas)
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                      download_name="datos_filtrados.csv")


@app.route("/api/descargar/resultados", methods=["GET"])
def descargar_resultados():
    datos = ULTIMO_RESULTADO.get("datos")
    if not datos or not datos["registros"]:
        return jsonify({"error": "Todavía no hay resultados generados. Entrena un modelo primero."}), 400

    buffer = _escribir_csv_en_memoria(datos["registros"], datos["columnas"])
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                      download_name="resultados_clustering.csv")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
