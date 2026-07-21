document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('upload-btn');
    const demoBtn = document.getElementById('demo-btn');
    const fileInput = document.getElementById('file-input');
    const toastContainer = document.getElementById('toast-container');
    
    const filterSection = document.getElementById('explore');
    const trainSection = document.getElementById('train');
    const resultsSection = document.getElementById('results');
    
    const categoryColSelect = document.getElementById('category-col');
    const categoryValSelect = document.getElementById('category-val');
    const applyFilterBtn = document.getElementById('apply-filter-btn');
    const downloadFilteredBtn = document.getElementById('download-filtered-btn');
    
    const algoCards = document.querySelectorAll('.algo-option');
    const linkageGroup = document.getElementById('linkage-group');
    const featuresCheckboxes = document.getElementById('features-checkboxes');
    const trainBtn = document.getElementById('train-btn');
    const trainSpinner = document.getElementById('train-spinner');
    
    // Novedad: Elementos del Modal de Estadísticas
    const statsBtn = document.getElementById('stats-btn');
    const statsModal = document.getElementById('stats-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modalVarSelect = document.getElementById('modal-var-select');
    const numericStatsGrid = document.getElementById('numeric-stats-grid');
    let descriptiveStatsData = null;
    let varChartInstance = null;
    let radarChartInstance = null;
    
    let currentAlgorithm = 'kmeans';
    let availableColumns = [];
    let chartInstance = null; // To keep track of Chart.js
    
    // Modern Toast Notification System
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'ℹ️';
        if (type === 'success') icon = '✅';
        if (type === 'error') icon = '❌';
        
        toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s reverse forwards';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    uploadBtn.addEventListener('click', async () => {
        if (!fileInput.files[0]) {
            showToast('No file selected.', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            showToast('Ingesting data...', 'info');
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (res.ok) {
                showToast(`Successfully ingested ${data.rows} records.`, 'success');
                availableColumns = data.columns;
                onDataLoaded();
            } else {
                showToast(`Error: ${data.error}`, 'error');
            }
        } catch (e) {
            showToast('Connection failed to backend.', 'error');
        }
    });

    demoBtn.addEventListener('click', () => {
        showToast('Please select "data/perfil_zodiacal_ejemplo.csv" from your folders to test.', 'info');
    });

    function onDataLoaded() {
        filterSection.style.display = 'block';
        trainSection.style.display = 'block';
        resultsSection.style.display = 'none'; // hide previous results
        statsBtn.style.display = 'inline-block'; // mostrar botón de stats
        
        categoryColSelect.innerHTML = '<option value="">-- Ninguno --</option>';
        availableColumns.forEach(col => {
            if (!['Edad', 'Energía'].includes(col)) {
                categoryColSelect.innerHTML += `<option value="${col}">${col}</option>`;
            }
        });
        
        featuresCheckboxes.innerHTML = '';
        availableColumns.forEach(col => {
            if (col !== 'Signo' && col !== 'Elemento') {
                featuresCheckboxes.innerHTML += `
                    <label>
                        <input type="checkbox" value="${col}" checked> ${col}
                    </label>
                `;
            }
        });
        
        loadTableData();
        loadStats();
    }
    
    categoryColSelect.addEventListener('change', async () => {
        const col = categoryColSelect.value;
        if (!col) {
            categoryValSelect.innerHTML = '<option value="">-- Todos --</option>';
            return;
        }
        
        const res = await fetch(`/categories?category_col=${col}`);
        const data = await res.json();
        
        categoryValSelect.innerHTML = '<option value="">-- Todos --</option>';
        if (data.values) {
            data.values.forEach(val => {
                categoryValSelect.innerHTML += `<option value="${val}">${val}</option>`;
            });
        }
    });
    
    applyFilterBtn.addEventListener('click', () => {
        loadTableData();
        loadStats();
        showToast('Filtros aplicados', 'success');
    });
    
    async function loadTableData() {
        const col = categoryColSelect.value;
        const val = categoryValSelect.value;
        
        let url = '/data';
        if (col && val) url += `?category_col=${col}&category_val=${val}`;
        
        const res = await fetch(url);
        const data = await res.json();
        
        if (data.data && data.data.length > 0) {
            renderTable('data-table', 'table-head', 'table-body', data.data);
        }
    }
    
    async function loadStats() {
        const col = categoryColSelect.value;
        const val = categoryValSelect.value;
        
        const checkboxes = document.querySelectorAll('#features-checkboxes input:checked');
        const features = Array.from(checkboxes).map(cb => cb.value);
        
        const body = { features };
        if (col && val) {
            body.category_col = col;
            body.category_val = val;
        }
        
        const res = await fetch('/stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const data = await res.json();
        if (data.stats) {
            const container = document.getElementById('stats-container');
            container.innerHTML = '';
            for (const [feat, stats] of Object.entries(data.stats)) {
                // Ensure the stat is not NaN stringified
                const mean = stats.mean !== undefined && stats.mean !== null ? stats.mean.toFixed(2) : '-';
                container.innerHTML += `
                    <div class="stat-card">
                        <h4>${feat}</h4>
                        <div class="stat-metric">Mean <span>${mean}</span></div>
                        <div class="stat-metric">Min <span>${stats.min ?? '-'}</span></div>
                        <div class="stat-metric">Max <span>${stats.max ?? '-'}</span></div>
                    </div>
                `;
            }
        }
    }
    
    downloadFilteredBtn.addEventListener('click', () => {
        const col = categoryColSelect.value;
        const val = categoryValSelect.value;
        let url = '/download?type=filtered';
        if (col && val) url += `&category_col=${col}&category_val=${val}`;
        window.location.href = url;
    });

    // --- LÓGICA DE ESTADÍSTICA DESCRIPTIVA (EDA) ---
    statsBtn.addEventListener('click', async () => {
        try {
            showToast('Generando estadística descriptiva...', 'info');
            const res = await fetch('/descriptive_stats');
            const data = await res.json();
            
            if (res.ok) {
                descriptiveStatsData = data.stats;
                statsModal.classList.remove('hidden');
                
                // Llenar el selector con las variables numéricas
                modalVarSelect.innerHTML = '';
                Object.keys(descriptiveStatsData).forEach(col => {
                    modalVarSelect.innerHTML += `<option value="${col}">${col}</option>`;
                });
                
                // Mostrar datos de la primera variable
                if(Object.keys(descriptiveStatsData).length > 0) {
                    renderDescriptiveStats(Object.keys(descriptiveStatsData)[0]);
                }
                
                // Renderizar fórmulas de MathJax si existen
                if (window.MathJax) {
                    MathJax.typesetPromise();
                }
            } else {
                showToast(`Error EDA: ${data.error}`, 'error');
            }
        } catch (e) {
            showToast('Connection failed during EDA.', 'error');
        }
    });
    
    closeModalBtn.addEventListener('click', () => {
        statsModal.classList.add('hidden');
    });
    
    modalVarSelect.addEventListener('change', (e) => {
        renderDescriptiveStats(e.target.value);
    });
    
    function renderDescriptiveStats(colName) {
        const stats = descriptiveStatsData[colName];
        if (!stats) return;
        
        if (stats.type === 'categorical') {
            numericStatsGrid.style.display = 'none';
            renderDoughnutChart(colName, stats.counts);
        } else {
            numericStatsGrid.style.display = 'flex';
            document.getElementById('m-mean').textContent = stats.mean;
            document.getElementById('m-median').textContent = stats.median;
            document.getElementById('m-var').textContent = stats.variance;
            document.getElementById('m-std').textContent = stats.std_dev;
            document.getElementById('m-range').textContent = stats.range;
            document.getElementById('m-min').textContent = stats.min;
            document.getElementById('m-max').textContent = stats.max;
            renderHistogram(colName, stats.histogram);
        }
        
        renderGlobalRadar();
    }
    
    function renderHistogram(colName, histogramData) {
        if (varChartInstance) varChartInstance.destroy();
        
        const ctx = document.getElementById('varChart').getContext('2d');
        const labels = histogramData.bins.slice(0, -1).map((b, i) => `${b.toFixed(2)} - ${histogramData.bins[i+1].toFixed(2)}`);
        
        varChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: `Frecuencia`,
                    data: histogramData.counts,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Histograma: ${colName}`, color: 'white', font: { size: 14 } },
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    function renderDoughnutChart(colName, countsData) {
        if (varChartInstance) varChartInstance.destroy();
        
        const ctx = document.getElementById('varChart').getContext('2d');
        const labels = Object.keys(countsData);
        const data = Object.values(countsData);
        
        varChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9'],
                    borderColor: 'transparent'
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Distribución: ${colName}`, color: 'white', font: { size: 14 } },
                    legend: { position: 'right', labels: { color: '#94a3b8' } }
                }
            }
        });
    }

    function renderGlobalRadar() {
        if (radarChartInstance) radarChartInstance.destroy();
        
        const numericCols = Object.keys(descriptiveStatsData).filter(k => descriptiveStatsData[k].type === 'numeric');
        if (numericCols.length === 0) return;
        
        // Normalize means to 0-1 scale for comparison
        let means = numericCols.map(col => descriptiveStatsData[col].mean);
        let maxMean = Math.max(...means) || 1;
        let normMeans = means.map(m => (m / maxMean) * 100);

        const ctx = document.getElementById('radarChart').getContext('2d');
        radarChartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: numericCols,
                datasets: [{
                    label: 'Perfil Global (Medias relativas)',
                    data: normMeans,
                    backgroundColor: 'rgba(99, 102, 241, 0.4)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#6366f1'
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: 'Radar de Atributos Numéricos', color: 'white', font: { size: 14 } },
                    legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255,255,255,0.1)' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 10 } },
                        ticks: { display: false }
                    }
                }
            }
        });
    }
    // -----------------------------------------------
    
    algoCards.forEach(card => {
        card.addEventListener('click', () => {
            algoCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            currentAlgorithm = card.dataset.algo;
            
            if (currentAlgorithm === 'agglomerative') {
                linkageGroup.style.display = 'flex';
            } else {
                linkageGroup.style.display = 'none';
            }
        });
    });
    
    trainBtn.addEventListener('click', async () => {
        const checkboxes = document.querySelectorAll('#features-checkboxes input:checked');
        const features = Array.from(checkboxes).map(cb => cb.value);
        
        if (features.length === 0) {
            showToast('Select at least 1 feature.', 'error');
            return;
        }
        
        const nClusters = document.getElementById('n-clusters').value;
        const linkage = document.getElementById('linkage-method').value;
        
        const col = categoryColSelect.value;
        const val = categoryValSelect.value;
        
        const body = {
            algorithm: currentAlgorithm,
            features: features,
            n_clusters: nClusters,
            linkage: linkage
        };
        if (col && val) {
            body.category_col = col;
            body.category_val = val;
        }
        
        try {
            // UI Loading state
            trainBtn.querySelector('.btn-text').textContent = 'Procesando tensores...';
            trainSpinner.classList.remove('hidden');
            trainBtn.disabled = true;
            
            const res = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            
            if (res.ok) {
                showToast('Model converged successfully! Visualizing results...', 'success');
                resultsSection.style.display = 'block';
                renderTable('results-table', 'results-table-head', 'results-table-body', data.results);
                
                // Draw Chart
                renderScatterPlot(data.results, features);
                
                // Scroll down
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                showToast(`Error de Entrenamiento: ${data.error}`, 'error');
            }
        } catch (e) {
            showToast('Connection failed during training.', 'error');
        } finally {
            // Reset UI
            trainBtn.querySelector('.btn-text').textContent = 'Iniciar Entrenamiento de IA';
            trainSpinner.classList.add('hidden');
            trainBtn.disabled = false;
        }
    });
    
    document.getElementById('download-results-btn').addEventListener('click', () => {
        window.location.href = '/download?type=results';
    });
    
    function renderTable(tableId, headId, bodyId, data) {
        if (!data || data.length === 0) return;
        
        const thead = document.getElementById(headId);
        const tbody = document.getElementById(bodyId);
        
        const cols = Object.keys(data[0]);
        thead.innerHTML = cols.map(c => `<th>${c}</th>`).join('');
        
        tbody.innerHTML = data.map(row => {
            return `<tr>${cols.map(c => `<td>${row[c] !== null ? row[c] : '-'}</td>`).join('')}</tr>`;
        }).join('');
    }
    
    function renderScatterPlot(data, features) {
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        const ctx = document.getElementById('clusterChart').getContext('2d');
        
        // We need at least 1 feature. If 1, we plot y=0. If >=2, we plot feat1 vs feat2.
        let f_x = features[0];
        let f_y = features.length > 1 ? features[1] : features[0];
        
        // Group data by Cluster
        const clusters = [...new Set(data.map(d => d.Cluster))].sort();
        
        // Generar colores neón para los clústeres
        const colors = [
            'rgba(99, 102, 241, 1)',   // Indigo
            'rgba(6, 182, 212, 1)',    // Cyan
            'rgba(168, 85, 247, 1)',   // Purple
            'rgba(245, 158, 11, 1)',   // Warning (Orange/Yellow)
            'rgba(16, 185, 129, 1)',   // Success (Green)
            'rgba(239, 68, 68, 1)',    // Danger (Red)
            'rgba(236, 72, 153, 1)'    // Pink
        ];
        
        const datasets = clusters.map((clusterId, index) => {
            const clusterData = data.filter(d => d.Cluster === clusterId).map(d => {
                return { x: d[f_x] || 0, y: features.length > 1 ? (d[f_y] || 0) : 0 };
            });
            
            const color = colors[index % colors.length];
            
            return {
                label: `Cluster ${clusterId}`,
                data: clusterData,
                backgroundColor: color,
                borderColor: color,
                pointRadius: 6,
                pointHoverRadius: 9,
            };
        });

        // Configurar tema oscuro de Chart.js
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";

        chartInstance = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: features.length > 1 ? `Dispersión 2D: ${f_x} vs ${f_y}` : `Dispersión 1D: ${f_x}`,
                        color: 'white',
                        font: { size: 16, weight: '600' }
                    },
                    legend: {
                        position: 'top',
                        labels: { color: 'white', usePointStyle: true, boxWidth: 10 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255,255,255,0.2)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: f_x, color: '#6366f1' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        title: { display: true, text: features.length > 1 ? f_y : 'N/A', color: '#a855f7' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    }
});
