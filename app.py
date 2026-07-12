from flask import Flask, request, render_template, jsonify, send_file
import os
import pandas as pd
import io
from ml_logic import train_kmeans, train_agglomerative, get_basic_stats

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

current_df = None
results_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        current_df = pd.read_csv(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'columns': current_df.columns.tolist(),
            'rows': len(current_df)
        })
    return jsonify({'error': 'Invalid file format. Please upload a CSV.'}), 400

@app.route('/data', methods=['GET'])
def get_data():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    category_col = request.args.get('category_col')
    category_val = request.args.get('category_val')
    
    df_to_return = current_df
    if category_col and category_val and category_col in current_df.columns:
        df_to_return = current_df[current_df[category_col] == category_val]
        
    return jsonify({
        'data': df_to_return.head(100).to_dict(orient='records'),
        'total_rows': len(df_to_return)
    })

@app.route('/categories', methods=['GET'])
def get_categories():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    col = request.args.get('category_col')
    if col in current_df.columns:
        unique_vals = current_df[col].dropna().unique().tolist()
        return jsonify({'values': unique_vals})
    return jsonify({'error': 'Column not found'}), 400

@app.route('/stats', methods=['POST'])
def get_stats():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json
    features = data.get('features', [])
    category_col = data.get('category_col')
    category_val = data.get('category_val')
    
    df_to_use = current_df
    if category_col and category_val and category_col in current_df.columns:
        df_to_use = current_df[current_df[category_col] == category_val]
        
    stats = get_basic_stats(df_to_use, features)
    return jsonify({'stats': stats})

@app.route('/train', methods=['POST'])
def train_model():
    global current_df, results_df
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json
    algorithm = data.get('algorithm', 'kmeans')
    features = data.get('features', [])
    n_clusters = int(data.get('n_clusters', 3))
    
    category_col = data.get('category_col')
    category_val = data.get('category_val')
    
    df_to_use = current_df.copy()
    if category_col and category_val and category_col in current_df.columns:
        df_to_use = df_to_use[df_to_use[category_col] == category_val]
        
    if len(features) == 0:
         return jsonify({'error': 'No features selected for training'}), 400
         
    try:
        # 1. Asegurar que las features sean numéricas (si el usuario mandó texto por error, lo forzamos a numérico y los textos se vuelven NaN)
        df_to_use[features] = df_to_use[features].apply(pd.to_numeric, errors='coerce')
        
        # 2. Rellenar los valores vacíos (NaN) con el promedio de la columna para que el algoritmo no falle
        df_to_use[features] = df_to_use[features].fillna(df_to_use[features].mean())
        
        # Si toda una columna es texto, su promedio es NaN, llenamos con 0
        df_to_use[features] = df_to_use[features].fillna(0)
        
        if algorithm == 'kmeans':
            clusters = train_kmeans(df_to_use, features, n_clusters, "kmeans_model")
        elif algorithm == 'agglomerative':
            linkage = data.get('linkage', 'ward')
            clusters = train_agglomerative(df_to_use, features, n_clusters, linkage, "agg_model")
        else:
            return jsonify({'error': 'Unknown algorithm'}), 400
            
        df_to_use['Cluster'] = clusters
        results_df = df_to_use
        
        # Convertimos NaN a None para que JSON no truene
        results_df = results_df.replace({pd.NA: None})
        import numpy as np
        results_df = results_df.replace({np.nan: None})
        
        return jsonify({
            'message': 'Model trained successfully',
            'results': results_df.head(100).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['GET'])
def download_data():
    global results_df, current_df
    download_type = request.args.get('type', 'filtered')
    
    df_to_download = None
    if download_type == 'results':
        df_to_download = results_df
    else:
        category_col = request.args.get('category_col')
        category_val = request.args.get('category_val')
        if current_df is not None:
            df_to_download = current_df
            if category_col and category_val and category_col in current_df.columns:
                df_to_download = current_df[current_df[category_col] == category_val]
                
    if df_to_download is None:
        return jsonify({'error': 'No data available to download'}), 400
        
    output = io.StringIO()
    df_to_download.to_csv(output, index=False)
    output.seek(0)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    
    filename = "resultados_clustering.csv" if download_type == 'results' else "datos_filtrados.csv"
    
    return send_file(
        mem,
        mimetype='text/csv',
        download_name=filename,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
