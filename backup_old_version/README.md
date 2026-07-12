# Perfilador Zodiacal — Aprendizaje No Supervisado

Aplicativo desarrollado para la actividad de Algoritmos No Supervisados
de la materia **Extracción de Conocimiento en Bases de Datos** (UTG).

Recolecta rasgos de personalidad de usuarios (energía, creatividad,
sociabilidad, estabilidad emocional, ambición) y aplica dos algoritmos
no supervisados para descubrir agrupamientos, sin usar el signo
zodiacal como variable de entrada del modelo:

- **K-Means**
- **Clusterización Jerárquica** (aglomerativa, con dendrograma)

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

Abre tu navegador en **http://localhost:5000**

## Estructura del proyecto

```
app_zodiacal/
├── app.py                 # Rutas Flask (API + vista principal)
├── estadisticas.py        # Funciones estadísticas propias (media, mediana, moda, etc.)
├── modelos.py              # Entrenamiento, guardado y carga de K-Means / Jerárquico
├── requirements.txt
├── data/
│   └── perfil_zodiacal_ejemplo.csv   # Dataset de ejemplo (50 registros)
├── models/                 # Aquí se guardan los modelos entrenados (.pkl)
├── uploads/                 # Aquí se guarda el CSV actualmente cargado
├── static/
│   ├── style.css
│   ├── app.js
│   └── dendrograma.png     # Se genera al entrenar el algoritmo jerárquico
└── templates/
    └── index.html
```

## Flujo de uso

1. **Cargar datos**: sube tu propio CSV o usa el dataset de ejemplo.
   El CSV debe incluir, como mínimo, las columnas:
   `elemento, signo_zodiacal, edad, energia, creatividad, sociabilidad, estabilidad_emocional, ambicion`
2. **Ver y filtrar los datos** por elemento o signo zodiacal.
3. **Revisar la estadística básica** (calculada con algoritmos propios,
   sin `pandas.describe()`).
4. **Entrenar el modelo**: elige K-Means o Jerárquico, el número de
   clústeres (k), los rasgos a considerar, y presiona "Iniciar
   aprendizaje". El modelo se guarda automáticamente en `models/` con
   `joblib` para reutilizarlo después.
5. **Ver resultados**: tabla con el clúster asignado a cada registro,
   métricas (coeficiente de silueta, inercia) y, en el caso del
   algoritmo jerárquico, el dendrograma generado.
6. **Descargar** tanto los datos filtrados como los resultados del
   clustering en formato CSV.

## Notas técnicas

- Los datos numéricos se normalizan con `StandardScaler` antes de
  entrenar, ya que las escalas de "edad" y los rasgos de personalidad
  (1–10) son distintas.
- El modelo de K-Means guardado puede reutilizarse para clasificar un
  usuario nuevo con la función `clasificar_nuevo_registro()` de
  `modelos.py` (la Clusterización Jerárquica no soporta `.predict()`
  sobre observaciones nuevas, es una limitación propia del algoritmo).
- El dataset de ejemplo fue generado sintéticamente con sesgos por
  elemento (Fuego, Tierra, Aire, Agua) para que los agrupamientos
  resultantes tengan sentido narrativo al presentarlos.
