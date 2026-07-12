const COLORES_CLUSTER = ["#D4AF6A", "#7C83FD", "#E8895A", "#7FC8D9", "#8FA86B", "#E06C9F"];
const COLORES_ELEMENTO = { Fuego: "#E8895A", Tierra: "#8FA86B", Aire: "#7FC8D9", Agua: "#6E8FD9" };

let algoritmoSeleccionado = "kmeans";

// ---------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------
async function llamarAPI(url, opciones = {}) {
  const respuesta = await fetch(url, opciones);
  const datos = await respuesta.json();
  if (!respuesta.ok) throw new Error(datos.error || "Ocurrió un error inesperado.");
  return datos;
}

function pintarTabla(tabla, registros) {
  const thead = tabla.querySelector("thead tr");
  const tbody = tabla.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (!registros || registros.length === 0) {
    tbody.innerHTML = `<tr><td style="color:var(--texto-tenue); padding:16px;">Sin registros para mostrar.</td></tr>`;
    return;
  }

  const columnas = Object.keys(registros[0]);
  columnas.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col.replace(/_/g, " ");
    thead.appendChild(th);
  });

  registros.forEach((fila) => {
    const tr = document.createElement("tr");
    columnas.forEach((col) => {
      const td = document.createElement("td");
      if (col === "cluster") {
        const color = COLORES_CLUSTER[Number(fila[col]) % COLORES_CLUSTER.length];
        td.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>${fila[col]}`;
      } else {
        td.textContent = fila[col];
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------
// 1) Carga de datos
// ---------------------------------------------------------------------
async function cargarArchivo() {
  const input = document.getElementById("input-archivo");
  const estado = document.getElementById("estado-carga");
  if (!input.files.length) {
    estado.textContent = "Selecciona un archivo CSV primero.";
    return;
  }
  const formData = new FormData();
  formData.append("archivo", input.files[0]);

  estado.textContent = "Cargando...";
  try {
    const datos = await llamarAPI("/api/cargar", { method: "POST", body: formData });
    estado.textContent = datos.mensaje;
    await refrescarTodo();
  } catch (e) {
    estado.textContent = "Error: " + e.message;
  }
}

async function cargarEjemplo() {
  const estado = document.getElementById("estado-carga");
  estado.textContent = "Cargando dataset de ejemplo...";
  try {
    const datos = await llamarAPI("/api/cargar-ejemplo", { method: "POST" });
    estado.textContent = datos.mensaje;
    await refrescarTodo();
  } catch (e) {
    estado.textContent = "Error: " + e.message;
  }
}

// ---------------------------------------------------------------------
// 2) Mostrar datos + 3) filtro
// ---------------------------------------------------------------------
function obtenerFiltroActual() {
  const columna = document.getElementById("select-columna-filtro").value;
  const valor = document.getElementById("select-valor-filtro").value;
  return { columna, valor };
}

async function refrescarCategorias() {
  const columna = document.getElementById("select-columna-filtro").value;
  const datos = await llamarAPI(`/api/categorias?columna=${columna}`);
  const select = document.getElementById("select-valor-filtro");
  select.innerHTML = `<option value="">Todos</option>`;
  datos.valores.forEach((valor) => {
    const opt = document.createElement("option");
    opt.value = valor;
    opt.textContent = valor;
    select.appendChild(opt);
  });
}

async function refrescarDatos() {
  const { columna, valor } = obtenerFiltroActual();
  const params = new URLSearchParams({ columna, valor });
  const datos = await llamarAPI(`/api/datos?${params}`);
  pintarTabla(document.getElementById("tabla-datos"), datos.registros);
  document.getElementById("contador-registros").textContent = `${datos.total} registro(s)`;
}

function descargarDatos() {
  const { columna, valor } = obtenerFiltroActual();
  const params = new URLSearchParams({ columna, valor });
  window.location.href = `/api/descargar/datos?${params}`;
}

// ---------------------------------------------------------------------
// 4) Estadística
// ---------------------------------------------------------------------
async function refrescarEstadisticas() {
  const { columna, valor } = obtenerFiltroActual();
  const params = new URLSearchParams({ columna, valor });
  let datos;
  try {
    datos = await llamarAPI(`/api/estadisticas?${params}`);
  } catch (e) {
    document.getElementById("tarjetas-estadisticas").innerHTML = "";
    document.getElementById("frecuencia-elemento").innerHTML = "";
    return;
  }

  const contenedor = document.getElementById("tarjetas-estadisticas");
  contenedor.innerHTML = "";
  Object.entries(datos.resumen_numerico).forEach(([columna, valores]) => {
    const div = document.createElement("div");
    div.className = "tarjeta-stat";
    div.innerHTML = `
      <h4>${columna.replace(/_/g, " ")}</h4>
      <dl>
        <div><span>media</span><span class="valor">${valores.media}</span></div>
        <div><span>mediana</span><span class="valor">${valores.mediana}</span></div>
        <div><span>moda</span><span class="valor">${valores.moda ?? "—"}</span></div>
        <div><span>desv. estándar</span><span class="valor">${valores.desviacion_estandar}</span></div>
        <div><span>mín – máx</span><span class="valor">${valores.minimo} – ${valores.maximo}</span></div>
        <div><span>Q1 / Q2 / Q3</span><span class="valor">${valores.cuartiles.Q1} / ${valores.cuartiles.Q2} / ${valores.cuartiles.Q3}</span></div>
      </dl>`;
    contenedor.appendChild(div);
  });

  const frecuencias = document.getElementById("frecuencia-elemento");
  frecuencias.innerHTML = `<p class="descripcion" style="margin-bottom:8px;">Distribución por elemento</p>`;
  Object.entries(datos.frecuencias_elemento).forEach(([elemento, info]) => {
    const color = COLORES_ELEMENTO[elemento] || "#D4AF6A";
    const fila = document.createElement("div");
    fila.className = "barra-frecuencia";
    fila.innerHTML = `
      <span class="etiqueta">${elemento}</span>
      <span class="pista"><span class="relleno" style="width:${info.porcentaje}%; background:${color};"></span></span>
      <span class="porcentaje">${info.porcentaje}%</span>`;
    frecuencias.appendChild(fila);
  });
}

// ---------------------------------------------------------------------
// 5) Entrenamiento
// ---------------------------------------------------------------------
function seleccionarAlgoritmo(algoritmo) {
  algoritmoSeleccionado = algoritmo;
  document.querySelectorAll(".tarjeta-algoritmo").forEach((btn) => {
    btn.classList.toggle("activa", btn.dataset.algoritmo === algoritmo);
  });
  document.getElementById("wrapper-enlace").style.display = algoritmo === "jerarquico" ? "flex" : "none";
  document.getElementById("input-nombre-modelo").value = `modelo_${algoritmo}`;
}

async function entrenarModelo() {
  const estado = document.getElementById("estado-entrenamiento");
  const columnas = Array.from(document.querySelectorAll(".columnas-entrenamiento input:checked")).map((i) => i.value);
  const { columna: columna_filtro, valor: valor_filtro } = obtenerFiltroActual();

  if (columnas.length < 2) {
    estado.textContent = "Selecciona al menos 2 rasgos para entrenar el modelo.";
    return;
  }

  const cuerpo = {
    algoritmo: algoritmoSeleccionado,
    k: Number(document.getElementById("input-k").value),
    columnas,
    columna_filtro,
    valor_filtro,
    nombre_modelo: document.getElementById("input-nombre-modelo").value || `modelo_${algoritmoSeleccionado}`,
    metodo_enlace: document.getElementById("select-enlace").value,
  };

  estado.textContent = "Entrenando modelo...";
  try {
    const resultado = await llamarAPI("/api/entrenar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cuerpo),
    });
    estado.textContent = `Modelo "${resultado.nombre_modelo}" entrenado y guardado correctamente.`;
    pintarResultados(resultado);
  } catch (e) {
    estado.textContent = "Error: " + e.message;
  }
}

// ---------------------------------------------------------------------
// 6) Resultados
// ---------------------------------------------------------------------
function pintarResultados(resultado) {
  const metricas = document.getElementById("metricas-resultado");
  metricas.innerHTML = "";

  const agregarMetrica = (etiqueta, valor) => {
    const div = document.createElement("div");
    div.className = "metrica";
    div.innerHTML = `<b>${valor}</b>${etiqueta}`;
    metricas.appendChild(div);
  };

  agregarMetrica("Algoritmo", resultado.algoritmo === "kmeans" ? "K-Means" : "Jerárquico");
  agregarMetrica("Clústeres (k)", resultado.k);
  if (resultado.silueta !== null && resultado.silueta !== undefined) {
    agregarMetrica("Coef. de silueta", resultado.silueta);
  }
  if (resultado.inercia !== undefined) {
    agregarMetrica("Inercia", resultado.inercia);
  }

  const imagen = document.getElementById("imagen-dendrograma");
  if (resultado.dendrograma_url) {
    imagen.src = resultado.dendrograma_url + "?t=" + Date.now();
    imagen.style.display = "block";
  } else {
    imagen.style.display = "none";
  }

  pintarTabla(document.getElementById("tabla-resultados"), resultado.registros_resultado);
}

function descargarResultados() {
  window.location.href = "/api/descargar/resultados";
}

// ---------------------------------------------------------------------
// Orquestación general
// ---------------------------------------------------------------------
async function refrescarTodo() {
  await refrescarCategorias();
  await refrescarDatos();
  await refrescarEstadisticas();
}

document.getElementById("btn-cargar").addEventListener("click", cargarArchivo);
document.getElementById("btn-ejemplo").addEventListener("click", cargarEjemplo);
document.getElementById("select-columna-filtro").addEventListener("change", async () => {
  await refrescarCategorias();
  await refrescarDatos();
  await refrescarEstadisticas();
});
document.getElementById("select-valor-filtro").addEventListener("change", async () => {
  await refrescarDatos();
  await refrescarEstadisticas();
});
document.getElementById("btn-limpiar-filtro").addEventListener("click", async () => {
  document.getElementById("select-valor-filtro").value = "";
  await refrescarDatos();
  await refrescarEstadisticas();
});
document.getElementById("btn-descargar-datos").addEventListener("click", descargarDatos);
document.getElementById("btn-entrenar").addEventListener("click", entrenarModelo);
document.getElementById("btn-descargar-resultados").addEventListener("click", descargarResultados);
document.querySelectorAll(".tarjeta-algoritmo").forEach((btn) => {
  btn.addEventListener("click", () => seleccionarAlgoritmo(btn.dataset.algoritmo));
});

// Carga inicial silenciosa (por si ya existe un dataset previamente cargado)
refrescarTodo();
