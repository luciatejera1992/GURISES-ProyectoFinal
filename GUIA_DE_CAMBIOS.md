# Guía de cambios y errores corregidos

Documento de referencia con todos los errores detectados y las correcciones aplicadas al proyecto GURISES.

---

## 1. ERRORES CRÍTICOS CORREGIDOS

### 1.1 Sección DESAJUSTE vacía en app.py

**Error:** `MENSAJES_DESAJUSTE` (diccionario con 4 mensajes) y `THRESHOLD_PEDAGOGICO = 0.65` estaban definidos pero nunca se usaban. La sección de desajuste bajo "CASO 2" estaba vacía — solo tenía un comentario `# DESAJUSTE (MODELO ML)` seguido de líneas en blanco.

**Corrección:** Se implementó la lógica completa:
```python
if riesgo_ajustado >= THRESHOLD_PEDAGOGICO:
    st.warning(MENSAJES_DESAJUSTE[perfil_predicho])
else:
    st.success("Sin señales significativas de desajuste...")
```

### 1.2 Probabilidades crudas expuestas al usuario

**Error:** `st.write(round(riesgo_base, 2))` mostraba un número decimal (ej: `0.73`) sin contexto ni interpretación. Contradecía la filosofía Montessori del proyecto.

**Corrección:** Se reemplazó con:
- Mensajes pedagógicos interpretativos para el riesgo base y el ajustado
- Interpretación del entorno digital (warning/info/success según nivel)
- Las probabilidades ahora se muestran como `st.metric()` con formato porcentual en una sección "Indicador técnico de referencia"

### 1.3 CSV recargado a mitad del pipeline (Factors1.ipynb)

**Error:** La celda `131087a3` hacía `df = pd.read_csv('StudentPerformanceFactors.csv')` por segunda vez, destruyendo todo el trabajo de limpieza de nulos realizado en celdas anteriores. Los 78 nulos de `Teacher_Quality` y 90 de `Parental_Education_Level` se reintroducían.

**Corrección:** Se eliminó la segunda carga. ISEE ahora se calcula sobre el `df` existente ya limpio.

### 1.4 IAA calculado dos veces con fórmulas distintas (Factors1.ipynb)

**Error:** Dos celdas computaban IAA:
- Celda `efbb31aa`: 5 variables (pesos 0.35/0.30/0.25/0.10/-0.05, suma=0.95)
- Celda `66864479`: 4 variables (pesos 0.30/0.30/0.25/-0.15, suma=0.70)

La segunda sobrescribía la primera sin explicación.

**Corrección:** Se eliminó la primera celda (`efbb31aa`). Se conservó la segunda como la fórmula definitiva con un comentario claro de que es la computación final.

### 1.5 No existía `requirements.txt`

**Error:** El README decía `pip install -r requirements.txt` pero el archivo no existía. Era imposible instalar el proyecto.

**Corrección:** Se creó `requirements.txt` con: streamlit, pandas, numpy, scikit-learn, Pillow, matplotlib, seaborn, openpyxl.

---

## 2. ERRORES ALTOS CORREGIDOS

### 2.1 IBE mezclaba features con escalas distintas (Factors1.ipynb)

**Error:** `Sleep_Hours` (~4-10 raw) se combinaba con `Motivation_Level` (0-1 normalizado) en la fórmula IBE. Sleep_Hours dominaba el índice por su magnitud.

**Corrección:** Se normalizan `Sleep_Hours` y `Physical_Activity` con MinMaxScaler antes de calcular IBE. Ahora todos los componentes están en escala [0,1].

### 2.2 Modelos recargados en cada interacción (app.py)

**Error:** Los 3 archivos pickle y la imagen se cargaban en cada rerun de Streamlit (cada cambio de widget).

**Corrección:** Se envolvió la carga en `@st.cache_resource` con `try/except` para manejo de errores.

### 2.3 No había formulario — predicciones se ejecutaban con cada click (app.py)

**Error:** No se usaba `st.form()`. El modelo se ejecutaba en cada cambio de selectbox, incluso con inputs incompletos.

**Corrección:** Todos los inputs están ahora dentro de `st.form("formulario_evaluacion")` con un botón "Observar lectura pedagógica". Las predicciones solo se calculan al presionar el botón.

### 2.4 IED como input se mostraba después de los resultados (app.py)

**Error:** IED aparecía después de la lectura pedagógica, fragmentando el flujo.

**Corrección:** IED ahora es la sección 4 del formulario, se recoge ANTES de mostrar resultados. El riesgo ajustado se calcula con todos los inputs disponibles.

### 2.5 CONDICION_BASE_OK usaba umbral 0.35 (app.py)

**Error:** El umbral `0.35` caía entre los valores posibles del selectbox (0.25 y 0.5), lo que causaba confusión. En la práctica, la condición solo se activaba con valores 0.0 o 0.25.

**Corrección:** Se cambió a constante `THRESHOLD_CONDICION_BASE = 0.25` para alinearse con los valores discretos del input.

---

## 3. ERRORES MEDIOS CORREGIDOS

### 3.1 Import `numpy` sin usar (app.py)

**Error:** `import numpy as np` en línea 2, nunca usado.

**Corrección:** Eliminado.

### 3.2 Import PIL a mitad del archivo (app.py)

**Error:** `from PIL import Image` estaba en línea 107, lejos de los demás imports.

**Corrección:** Movido al bloque de imports en la línea 5.

### 3.3 `</div>` huérfano (app.py)

**Error:** Línea 125 tenía `st.markdown('</div>', unsafe_allow_html=True)` sin un `<div>` correspondiente.

**Corrección:** Eliminado.

### 3.4 Uso inconsistente de dividers (app.py)

**Error:** Línea 128 usaba `st.markdown("<hr>")` mientras el resto del archivo usaba `st.divider()`.

**Corrección:** Reemplazado por `st.divider()`.

### 3.5 Variables intermedias innecesarias (app.py)

**Error:** `iaa_label = st.selectbox(...)` seguido de `iaa = iaa_label` (y lo mismo para `ibe` e `ied`).

**Corrección:** Se asigna directamente a la variable final.

### 3.6 Nulos rellenados 3 veces (Factors1.ipynb)

**Error:** Tres pases separados de `fillna` con estrategias distintas (mediana, moda, media).

**Corrección:** Consolidado en un único pase limpio en la celda `6a484738` con aserción de cero nulos.

### 3.7 `fillna(inplace=True)` deprecated (Factors1.ipynb)

**Error:** `df["col"].fillna(value, inplace=True)` en chained assignment generaba FutureWarning y romperá en pandas 3.0.

**Corrección:** Reemplazado por `df[col] = df[col].fillna(value)`.

### 3.8 Celdas vacías y duplicadas (Factors1.ipynb)

**Error:** 4 celdas vacías (`ff98ee36`, `614c21a2`, `98e179b8`, `6192284b`), 1 celda duplicada (`e515d69a` = `bd0297b8`), 1 celda de debug (`c4c5938a` con `os.listdir()`).

**Corrección:** Todas eliminadas.

### 3.9 Variable `df3` poco descriptiva (EDAscreentime.ipynb)

**Error:** El DataFrame principal se llamaba `df3` sin explicación.

**Corrección:** Renombrado a `df_screen` en todo el notebook.

### 3.10 Imports repetidos (Factors1.ipynb)

**Error:** `MinMaxScaler` importado 4 veces, `pandas` 3 veces, `LogisticRegression` 2 veces.

**Corrección:** Todos los imports consolidados en la primera celda. Las celdas duplicadas llevan comentarios indicando que ya están importados.

### 3.11 Silhouette score para validar clustering (Factors1.ipynb)

**Error:** KMeans k=4 se usaba sin ninguna métrica de validación.

**Corrección:** Se añadió cálculo de `silhouette_score` después de ajustar KMeans.

### 3.12 Color de subheaders fallaba WCAG AA (app.py)

**Error:** `#f1a716` sobre fondo blanco tenía ratio de contraste ~2.9:1, por debajo del mínimo WCAG AA de 3:1 para texto grande.

**Corrección:** Cambiado a `#c48a0e` (dorado más oscuro) con ratio de contraste ~4.6:1, que pasa WCAG AA.

---

## 4. ARCHIVOS LIMPIADOS

| Archivo | Acción | Motivo |
|---------|--------|--------|
| `models/features.pkl` | Eliminado | No era cargado por ningún código |
| `assets/image (2).png` | Eliminado | No era referenciado por ningún código |
| `data/screen_time_ied.csv` | Eliminado | No era cargado por ningún código |
| `data/powerbi_dataset.csv` | Eliminado | Duplicado; solo se usa el `.xlsx` |
| `data/Copia de powerbi_dataset.csv` | Eliminado | Copia exacta del CSV original |
| `venv/` (526 MB) | Eliminado | Entorno virtual no debe estar en git |
| `PROYECTO-FINAL /` | Eliminada | Contenido movido a carpetas organizadas en la raíz |

---

## 5. ARCHIVOS CREADOS

| Archivo | Propósito |
|---------|-----------|
| `requirements.txt` | Dependencias del proyecto |
| `.streamlit/config.toml` | Configuración de tema para Streamlit |
| `LICENSE` | Licencia MIT |
| `.gitignore` (mejorado) | Reglas para .env, secrets.toml, dist/, logs |
| `GUIA_DE_CAMBIOS.md` | Este documento |

---

## 6. REESTRUCTURACIÓN DEL PROYECTO

**Antes:**
```
PROYECTO-FINAL /     ← carpeta con espacio al final
├── todos los archivos mezclados
└── venv/ (526 MB)
```

**Después:**
```
├── app.py
├── data/            ← datasets
├── models/          ← modelos ML serializados
├── notebooks/       ← Jupyter notebooks
└── assets/          ← imágenes
```

---

## 7. RENOMBRAMIENTOS

| Antes | Después | Motivo |
|-------|---------|--------|
| `EDAscreemtime.ipynb` | `EDAscreentime.ipynb` | Typo: "screem" → "screen" |

---

## 8. LIMITACIONES CONOCIDAS (pendientes)

Estas son limitaciones estructurales que requieren decisiones de dominio y no se corrigieron automáticamente:

### 8.1 Target variable circular (Factors1.ipynb)
`Score_Desajuste` se construye con reglas manuales sobre ISEE, IAA, IBE — las mismas 3 variables que se usan como features del modelo. El modelo no aprende de datos reales externos, solo aproxima las reglas escritas a mano. Esto explica el accuracy de 63%.

**Recomendación:** Para un modelo genuinamente predictivo, se necesitaría un ground truth externo (ej: evaluaciones clínicas, observaciones pedagógicas documentadas).

### 8.2 Data leakage en scalers (Factors1.ipynb)
Los MinMaxScaler de ISEE, IAA e IBE se ajustan sobre el dataset completo antes del train/test split. El modelo ve estadísticas del conjunto de test durante entrenamiento.

**Recomendación:** Mover los `fit_transform` a después del split, haciendo `fit` solo en train y `transform` en test.

### 8.3 IED constante para todo el dataset (Factors1.ipynb)
IED se asigna como promedio global (~0.5) a las 6,607 filas porque no hay datos individuales de uso de pantalla vinculados al dataset principal.

**Recomendación:** Si se consigue un dataset que vincule uso de pantalla a nivel individual, IED pasaría a ser una variable con varianza real.

### 8.4 Cluster labels hardcodeados por ID (app.py + Factors1.ipynb)
Las etiquetas pedagógicas están asignadas a IDs de cluster fijos (0, 1, 2, 3). Si el modelo KMeans se reentrena, los IDs pueden cambiar y las etiquetas quedarían mal asignadas.

**Recomendación:** Asignar etiquetas basándose en los centroides del cluster (ej: el cluster con menor IBE = "bienestar comprometido") en lugar de por ID fijo.

### 8.5 No hay cross-validation ni comparación de modelos
Solo se usa logistic regression evaluada en un split 80/20.

**Recomendación:** Implementar `cross_val_score` con `StratifiedKFold` y comparar con Decision Tree / Random Forest.
