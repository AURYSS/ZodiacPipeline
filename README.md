# Zodiacal Clustering Pipeline 🔮

Este proyecto corresponde al análisis no supervisado de encuestas zodiacales utilizando algoritmos de clustering (K-Means, DBSCAN, GMM) y procesamiento de lenguaje natural (NLP) con base de datos PostgreSQL.

---

## 🚀 Instrucciones de Ejecución

### Requisito Previo
Asegúrate de tener un servidor de **PostgreSQL** activo localmente con una base de datos llamada `zodiac`.

---

### 💻 macOS / Linux

Abra la terminal en la carpeta del proyecto y ejecute:

```bash
# 1. Crear entorno virtual con Python 3.12 (o superior estable)
python3.12 -m venv venv

# 2. Activar el entorno virtual
source venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Iniciar la aplicación web
streamlit run app.py
```

---

### 🪟 Windows (CMD o PowerShell)

Abra la consola (Símbolo del sistema o PowerShell) en la carpeta del proyecto y ejecute:

```cmd
:: 1. Crear entorno virtual
python -m venv venv

:: 2. Activar el entorno virtual
:: Si usa Símbolo del Sistema (CMD):
.\venv\Scripts\activate.bat

:: Si usa PowerShell:
.\venv\Scripts\Activate.ps1

:: 3. Instalar las dependencias
pip install -r requirements.txt

:: 4. Iniciar la aplicación web
streamlit run app.py
```

> **Nota para PowerShell en Windows**: Si al intentar activar el entorno le aparece un error de políticas de ejecución, ejecute primero:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` y vuelva a intentar la activación.

---

## 🤖 Entrenamiento de Modelos por Consola

Puedes entrenar y guardar los modelos directamente leyendo de la base de datos PostgreSQL ejecutando en la terminal (con el entorno virtual activo):

```bash
python train_models.py
```
