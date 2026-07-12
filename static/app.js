document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('upload-btn');
    const demoBtn = document.getElementById('demo-btn');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    
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
    const trainStatus = document.getElementById('train-status');
    
    let currentAlgorithm = 'kmeans';
    let availableColumns = [];
    
    function logMsg(el, msg, type = 'info') {
        el.textContent = msg;
        el.className = `log-msg ${type}`;
    }

    uploadBtn.addEventListener('click', async () => {
        if (!fileInput.files[0]) {
            logMsg(uploadStatus, 'ERR: No file selected.', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            logMsg(uploadStatus, 'Uploading file...', 'info');
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (res.ok) {
                logMsg(uploadStatus, `SUCCESS: Ingested ${data.rows} records.`, 'success');
                availableColumns = data.columns;
                onDataLoaded();
            } else {
                logMsg(uploadStatus, `ERR: ${data.error}`, 'error');
            }
        } catch (e) {
            logMsg(uploadStatus, 'ERR: Connection failed.', 'error');
        }
    });

    demoBtn.addEventListener('click', async () => {
        logMsg(uploadStatus, 'Select data/perfil_zodiacal_ejemplo.csv and upload.', 'info');
    });

    function onDataLoaded() {
        filterSection.style.display = 'block';
        trainSection.style.display = 'block';
        
        categoryColSelect.innerHTML = '<option value="">-- Ninguno --</option>';
        availableColumns.forEach(col => {
            if (!['Edad', 'Energía'].includes(col)) { // Basic heuristic
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
                container.innerHTML += `
                    <div class="stat-card">
                        <h4>${feat}</h4>
                        <div class="stat-metric">Mean <span>${stats.mean}</span></div>
                        <div class="stat-metric">Min <span>${stats.min}</span></div>
                        <div class="stat-metric">Max <span>${stats.max}</span></div>
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
    
    algoCards.forEach(card => {
        card.addEventListener('click', () => {
            algoCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            currentAlgorithm = card.dataset.algo;
            
            if (currentAlgorithm === 'agglomerative') {
                linkageGroup.style.display = 'block';
            } else {
                linkageGroup.style.display = 'none';
            }
        });
    });
    
    trainBtn.addEventListener('click', async () => {
        const checkboxes = document.querySelectorAll('#features-checkboxes input:checked');
        const features = Array.from(checkboxes).map(cb => cb.value);
        
        if (features.length === 0) {
            logMsg(trainStatus, 'ERR: Select >= 1 feature.', 'error');
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
            logMsg(trainStatus, 'Running model.fit_predict()...', 'info');
            trainBtn.disabled = true;
            
            const res = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            
            if (res.ok) {
                logMsg(trainStatus, 'SUCCESS: Model trained. Inference complete.', 'success');
                resultsSection.style.display = 'block';
                renderTable('results-table', 'results-table-head', 'results-table-body', data.results);
            } else {
                logMsg(trainStatus, `ERR: ${data.error}`, 'error');
            }
        } catch (e) {
            logMsg(trainStatus, 'ERR: Connection failed.', 'error');
        } finally {
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
            return `<tr>${cols.map(c => `<td>${row[c]}</td>`).join('')}</tr>`;
        }).join('');
    }
});
