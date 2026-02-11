# ============================================
# IMPORTACIONES
# ============================================

# Framework de la app
import streamlit as st

# Manipulación de datos
import pandas as pd
import numpy as np

# Visualización
import plotly.express as px
import plotly.graph_objects as go

# Preprocesamiento (solo para clustering no supervisado)
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Gestión de archivos y rutas
from pathlib import Path

# Imágenes (logo / marca)
from PIL import Image


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="GURISES — Un Caracol Montessori",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================
# UMBRALES PEDAGÓGICOS (REFERENCIAS ORIENTATIVAS)
# ============================================

THRESHOLD_PEDAGOGICO = 0.65
# Punto de atención pedagógica.
# Indica necesidad de observación más cuidadosa.
# No constituye diagnóstico ni clasificación.

THRESHOLD_CONDICION_BASE = 0.25
# Umbral mínimo de condición de base (bienestar).
# Si no se alcanza, se prioriza el acompañamiento
# antes de interpretar cualquier perfil educativo.


# =========================================================
# RUTAS
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "studentperformancefactors.csv"

# ============================================
# CABECERA Y LOGO
# ============================================

logo_path = BASE_DIR / "assets" / "logo.png"

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(logo_path, width=100)
with col_title:
    st.markdown(
        "<h2 style='margin-bottom:0;color:#4763a2;'>GURISES</h2>"
        "<p style='margin-top:0;color:#555;font-size:1.1em;'>"
        "Un Caracol Montessori · Herramienta de lectura pedagógica</p>",
        unsafe_allow_html=True,
    )


# ============================================
# PALETA DE COLORES - IDENTIDAD VISUAL
# ============================================

# Colores principales de marca
COLOR_NAVY  = "#4763a2"   # estructura, confianza
COLOR_GOLD  = "#c48a0e"   # valor, potencial
COLOR_LIGHT = "#f9fafc"   # entorno preparado
COLOR_WHITE = "#ffffff"   # claridad

# Paleta para visualizaciones
PLOTLY_COLORS = [
    "#4763a2",  # navy
    "#c48a0e",  # gold
    "#6ba368",  # verde equilibrio
    "#d4615e"   # terracota atención pedagógica
]
# ============================================
# ESTILO VISUAL (CSS SUAVE)
# ============================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COLOR_LIGHT};
            color: {COLOR_NAVY};
        }}
        h1, h2, h3 {{
            color: {COLOR_NAVY};
        }}
        .stButton > button {{
            background-color: {COLOR_NAVY};
            color: {COLOR_WHITE};
            border-radius: 6px;
        }}
        .stButton > button:hover {{
            background-color: {COLOR_GOLD};
            color: {COLOR_WHITE};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# — CARGA DE DATOS BASE
# ============================================

DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "studentperformancefactors.csv"

if not DATA_FILE.exists():
    st.error(
        "No se encuentra el archivo de datos base "
        "`studentperformancefactors.csv` en la carpeta /data."
    )
    st.stop()

try:
    df_raw = pd.read_csv(DATA_FILE)
except Exception:
    st.error("Error al cargar el archivo de datos base.")
    st.stop()


# Validación de columnas requeridas para el análisis y la evaluación pedagógica.
COLUMNAS_REQUERIDAS = [
    "Hours_Studied",
    "Attendance",
    "Tutoring_Sessions",
    "Sleep_Hours",
    "Physical_Activity",
    "Parental_Involvement",
    "Access_to_Resources",
    "Teacher_Quality",
    "Motivation_Level",
    "Peer_Influence",
    "School_Type"
]


faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df_raw.columns]

if faltantes:
    st.error(
        f"El dataset no contiene las columnas requeridas: {faltantes}"
    )
    st.stop()


# ============================================
# LIMPIEZA BÁSICA DEL DATASET
# ============================================

# Copia de trabajo (preservamos datos originales)
df = df_raw.copy()

# Eliminar filas completamente vacías
df.dropna(how="all", inplace=True)

# Asegurar tipos numéricos donde corresponde
COLUMNAS_NUMERICAS = [
    "Hours_Studied",
    "Sleep_Hours",
    "Attendance"
]

for col in COLUMNAS_NUMERICAS:
    df[col] = pd.to_numeric(df[col], errors="coerce")





# =========================================================
# CONSTRUCCIÓN DE ÍNDICES PEDAGÓGICOS
# =========================================================

from sklearn.preprocessing import MinMaxScaler

def construir_indices_pedagogicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye índices pedagógicos Montessori a partir de variables observables.
    Traduce observaciones cualitativas a escalas ordinales explícitas.
    """

    df = df.copy()

    # ----------------------------
    # Codificación ordinal pedagógica
    # ----------------------------
    MAPA_ORDINAL = {
        "Low": 0.33,
        "Medium": 0.66,
        "High": 1.0
    }

    COLUMNAS_ORDINALES = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Teacher_Quality",
        "Motivation_Level",
        "Peer_Influence"
    ]

    for col in COLUMNAS_ORDINALES:
        df[col] = df[col].map(MAPA_ORDINAL)
        df[col] = df[col].fillna(0.5)  # valor neutro pedagógico

    # ----------------------------
    # Escalado de variables numéricas reales
    # ----------------------------
    COLUMNAS_NUMERICAS = [
        "Hours_Studied",
        "Attendance",
        "Tutoring_Sessions",
        "Sleep_Hours",
        "Physical_Activity"
    ]

    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].fillna(
        df[COLUMNAS_NUMERICAS].median()
    )

    scaler = MinMaxScaler()
    df[COLUMNAS_NUMERICAS] = scaler.fit_transform(df[COLUMNAS_NUMERICAS])

    # ----------------------------
    # Codificación School Type
    # ----------------------------
    df["School_Type_Num"] = df["School_Type"].map({
        "Public": 0.7,
        "Private": 1.0
    })

    # ----------------------------
    # ISEE — indice de soporte del entorno educativo
    # ----------------------------
    df["ISEE"] = (
        df["Parental_Involvement"] * 0.25 +
        df["Access_to_Resources"] * 0.25 +
        df["School_Type_Num"] * 0.20 +
        df["Teacher_Quality"] * 0.30
    )

    # ----------------------------
    # IAA — Autonomía y autodisciplina
    # ----------------------------
    df["IAA"] = (
        df["Hours_Studied"] * 0.30 +
        df["Attendance"] * 0.30 +
        df["Motivation_Level"] * 0.25 +
        df["Tutoring_Sessions"] * 0.15
    )

    # ----------------------------
    # IBE — Indice de bienestar y equilibrio
    # ----------------------------
    df["IBE"] = (
        df["Sleep_Hours"] * 0.35 +
        df["Physical_Activity"] * 0.25 +
        df["Motivation_Level"] * 0.20 +
        df["Peer_Influence"] * 0.20
    )

    return df

# ============================================
# APLICACIÓN DE ÍNDICES PEDAGÓGICOS
# ============================================

df = construir_indices_pedagogicos(df)

# ============================================
#  ÍNDICE DE OBSERVACIÓN EDUCATIVA
# ============================================

# Pesos pedagógicos (suman 1)
W_IBE = 0.40   # Bienestar integral (condición habilitante)
W_ISEE = 0.30  # Entorno preparado
W_IAA = 0.30   # Autonomía y autodisciplina

df["indice_observacion_educativa"] = (
    W_IBE * (1 - df["IBE"]) +
    W_ISEE * (1 - df["ISEE"]) +
    W_IAA * (1 - df["IAA"])
).clip(0, 1)

# Nota:
# El índice es continuo y orientativo.
# Valores más altos indican mayor necesidad de observación pedagógica,
# no riesgo ni diagnóstico.

# ============================================
# BLINDAJE FINAL ANTES DE CLUSTERING
# ============================================

for col in ["ISEE", "IAA", "IBE"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ============================================
# CLUSTERING Y PERFILES
# ============================================

X_cluster = df[["ISEE", "IAA", "IBE"]]
X_cluster_scaled = MinMaxScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X_cluster_scaled)

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["ISEE", "IAA", "IBE"])

cluster_labels = {}
for idx, row in centroids.iterrows():
    if row["IBE"] < 0.4:
        cluster_labels[idx] = "Perfil con bienestar comprometido"
    elif row["ISEE"] > 0.6 and row["IAA"] > 0.6:
        cluster_labels[idx] = "Perfil educativo equilibrado"
    elif row["ISEE"] > row["IAA"]:
        cluster_labels[idx] = "Entorno favorable con autonomía en construcción"
    else:
        cluster_labels[idx] = "Perfil con autonomía alta y entorno exigente"

df["Condicion_Base_OK"] = df["IBE"] >= THRESHOLD_CONDICION_BASE
df["Perfil_Final"] = "Perfil educativo equilibrado"
df.loc[~df["Condicion_Base_OK"], "Perfil_Final"] = "Condición de base comprometida"
df.loc[df["Condicion_Base_OK"], "Perfil_Final"] = df["cluster_id"].map(cluster_labels)






PERFILES_DISPONIBLES = [
    "Perfil educativo equilibrado",
    "Entorno favorable con autonomía en construcción",
    "Perfil con autonomía alta y entorno exigente",
    "Perfil con bienestar comprometido",
    "Condición de base comprometida"
]


# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================

tab_inicio, tab_datos, tab_indices, tab_perfiles, tab_metodo = st.tabs([
    " Inicio",
    " Datos y contexto",
    " Índices pedagógicos",
    " Perfiles educativos",
    " Metodología"
])

# ============================================
# MENSAJES PEDAGÓGICOS POR PERFIL
# ============================================

MENSAJES_PERFIL = {
    "Perfil educativo equilibrado": """
    ### Perfil educativo equilibrado

    El entorno, la autonomía y el bienestar se encuentran en armonía,
    favoreciendo un desarrollo fluido y autónomo.

    **Orientación pedagógica**
    - Mantener la coherencia del ambiente.
    - Evitar intervenciones innecesarias.
    - Observar con confianza los procesos naturales de aprendizaje.

    *Fundamento Montessori*:
    > “Cuando el ambiente es adecuado, el niño trabaja y se construye a sí mismo.”
    """,

    "Entorno favorable con autonomía en construcción": """
    ### Entorno favorable con autonomía en construcción

    El entorno ofrece buenas condiciones, mientras que la autonomía
    se encuentra aún en proceso de consolidación.

    **Orientación pedagógica**
    - Revisar el grado de ayuda ofrecida.
    - Aumentar oportunidades reales de elección.
    - Permitir tiempo suficiente para el error y la repetición.

    *Fundamento Montessori*:
    > “La ayuda innecesaria es un obstáculo para el desarrollo.”
    """,

    "Perfil con autonomía alta y entorno exigente": """
    ### Autonomía alta con entorno exigente

    La autonomía está bien desarrollada, pero el entorno puede estar
    resultando excesivamente demandante o estructurado.

    **Orientación pedagógica**
    - Simplificar el ambiente.
    - Reducir estímulos y expectativas externas.
    - Priorizar el ritmo individual.

    *Fundamento Montessori*:
    > “El desarrollo necesita tiempo y condiciones favorables.”
    """,

    "Perfil con bienestar comprometido": """
    ### Bienestar comprometido

    El bienestar físico y/o emocional se encuentra afectado,
    lo que limita los procesos de aprendizaje profundo.

    **Prioridad pedagógica**
    - Restablecer calma y equilibrio emocional.
    - Reducir demandas y exigencias innecesarias.
    - Acompañar sin presionar.

    *Fundamento Montessori*:
    > “Sin equilibrio físico y emocional, el trabajo profundo no puede sostenerse.”
    """,

    "Condición de base comprometida": """
    ### Condición de base comprometida

    Antes de interpretar cualquier perfil educativo,
    es necesario atender las condiciones básicas de bienestar.

    **Prioridad pedagógica**
    - Garantizar seguridad, calma y cuidado.
    - Suspender expectativas de rendimiento.
    - Acompañar desde la presencia adulta.

    *Fundamento Montessori*:
    > “La paz es la base de la educación.”
    """
}


# ============================================

def mensaje_alerta_orientativa(valor_indice: float) -> str:
    """
    Devuelve un mensaje orientativo según el nivel del índice
    de observación educativa. No clasifica ni diagnostica.
    """
    if valor_indice < 0.33:
        return (
            "🟢 **Observación tranquila**\n\n"
            "El nivel de observación sugerido es bajo. "
            "Se recomienda continuar observando sin introducir cambios innecesarios."
        )
    elif valor_indice < THRESHOLD_PEDAGOGICO:
        return (
            "🟡 **Observación atenta**\n\n"
            "Puede ser útil observar con mayor atención la interacción "
            "entre el entorno, la autonomía y el bienestar."
        )
    else:
        return (
            "🟠 **Observación prioritaria**\n\n"
            "Se recomienda priorizar la observación pedagógica "
            "y revisar posibles ajustes del entorno antes de introducir nuevas exigencias."
        )




# =======================================================


# =============================================================
# PESTAÑA 1 — INICIO
# =============================================================
with tab_inicio:
    st.header("GURISES, DATOS, DESARROLLO Y EDUCACIÓN")

    st.markdown(
        """
        Esta aplicación ofrece una **lectura pedagógica orientativa**
        basada en principios Montessori.

        No evalúa, no diagnostica ni clasifica al niño.
        Su finalidad es **acompañar la observación educativa**
        y apoyar la adaptación consciente del entorno.

        La herramienta está diseñada para ser utilizada por
        familias, docentes y equipos educativos.
        """
    )





# =============================================================
# PESTAÑA 2 — DATOS Y CONTEXTO
# =============================================================

with tab_datos:
    st.subheader("Datos")

    st.markdown(
        """
        Los datos utilizados provienen de un **dataset educativo estructurado**
        que recoge información observacional sobre hábitos de estudio,
        entorno educativo y variables de bienestar.

        En esta sección se presenta el **contexto general de los datos**
        y su estructura, sin realizar interpretaciones pedagógicas.
        """
    )


    with st.expander("Estructura del dataset"):
        st.markdown(
            f"""
            - Número de registros: **{df_raw.shape[0]}**
            - Número de variables: **{df_raw.shape[1]}**
            """
        )
        st.dataframe(
            pd.DataFrame({
                "Variable": df_raw.columns,
                "Tipo de dato": df_raw.dtypes.astype(str)
            })
        )

    st.info(
        """
        **Nota metodológica**

        En esta etapa solo se realizan verificaciones estructurales
        y limpieza mínima de los datos.

        No se introducen interpretaciones pedagógicas ni conclusiones.
        Estas se desarrollan posteriormente a través de los índices educativos.
        """
    )


    col1, col2 = st.columns(2)
with col1: 
    st.subheader("¿Qué es Montessori?")
    st.markdown(
        """
        La pedagogía Montessori se fundamenta en la observación científica,
        y en la creación de un entorno preparado que favorezca el desarrollo natural.""")

    st.markdown("""Pilares de la pedagogía Montessori

**El niño:** protagonista activo de su propio desarrollo, guiado por sus ritmos internos y su capacidad natural de aprendizaje.

**El ambiente preparado:** espacio cuidadosamente diseñado para favorecer la autonomía, el orden y la exploración independiente.

**El adulto como guía:** observa, acompaña y ajusta el entorno sin interferir innecesariamente en el proceso del niño.

**Los materiales:** herramientas concretas y autocorrectivas que permiten aprender a través de la experiencia directa.

    """)
    with col2:
        st.image(
        "assets/1-María-Montessori.jpg",
        caption="María Montessori (Italia 1870-1952)"
    )


# =============================================================
# PESTAÑA 3 — ÍNDICES PEDAGÓGICOS
# =============================================================

with tab_indices:
    st.header("Índices pedagógicos")


    st.markdown(
    """

    A partir de los datos disponibles, se construyen **índices pedagógicos**
    que permiten una lectura educativa más integrada.

    Estos índices no miden rendimiento ni diagnostican,
    sino que **sintetizan patrones de observación**
    relacionados con el entorno, la autonomía y el bienestar.
    """
)
    st.markdown(
        """
        Los índices pedagógicos permiten una lectura integrada
        del entorno, la autonomía y el bienestar, en coherencia
        con la pedagogía Montessori.
        """
    )

    st.divider()

    idx_col1, idx_col2, idx_col3 = st.columns(3)
    with idx_col1:
        st.markdown("""
        **ISEE — Entorno preparado**

        Mide la calidad del ambiente educativo: orden, recursos, apoyo parental
        y calidad docente. Un entorno preparado facilita la autonomía y la concentración.
        """)
    with idx_col2:
        st.markdown("""
        **IAA — Autonomía y autodisciplina**

        Evalúa la capacidad del niño para iniciar y sostener actividades por cuenta propia,
        mantener el interés y depender menos de estímulos externos.
        """)
    with idx_col3:
        st.markdown("""
        **IBE — Bienestar y equilibrio**

        Refleja el estado físico, emocional y social del niño. El bienestar es condición
        indispensable para el aprendizaje profundo.
    """)



    with st.expander("Ver índices"):
        st.dataframe(
            df[["ISEE", "IAA", "IBE", "indice_observacion_educativa"]].head()
        )


    st.markdown(
    """
    ## Índice de observación educativa

    El **índice de observación educativa** integra distintas dimensiones
    del desarrollo para orientar la mirada pedagógica.

    Un valor más alto indica que puede ser útil **observar con mayor atención**
    cómo el entorno, la autonomía y el bienestar interactúan en el proceso educativo.

    Este índice **no evalúa ni diagnostica**; acompaña la observación y la adaptación del entorno.
    """
)

# =========================================================
# Ver índice de observación educativa

    with st.expander("Ver índice de observación educativa"):
        st.dataframe(
        df[["ISEE", "IAA", "IBE", "indice_observacion_educativa"]].head()
    )


# =============================================================
# PESTAÑA 4 — PERFILES EDUCATIVOS
# =============================================================

with tab_perfiles:
    st.header("Perfiles educativos")

    st.markdown(
        """
        Los perfiles educativos representan **patrones generales observados**
        en el conjunto de datos.

        No describen a un niño en particular, sino **configuraciones del entorno,
        la autonomía y el bienestar** que ayudan a orientar la observación pedagógica.
        """
    )

    # ---------------------------------------------------------
    # Selección de perfil (control principal de interacción)
    # ---------------------------------------------------------

    st.markdown("### Exploración pedagógica")

    perfil_seleccionado = st.radio(
        "Selecciona un perfil educativo",
        options=PERFILES_DISPONIBLES,
        horizontal=True
    )

    # Dataset filtrado por perfil
    df_perfil = df[df["Perfil_Final"] == perfil_seleccionado]

    # ---------------------------------------------------------
    # Visualización pedagógica del perfil (RADAR)
    # ---------------------------------------------------------


    perfil_media = {
        "Entorno (ISEE)": df_perfil["ISEE"].mean(),
        "Autonomía (IAA)": df_perfil["IAA"].mean(),
        "Bienestar (IBE)": df_perfil["IBE"].mean()
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=list(perfil_media.values()),
            theta=list(perfil_media.keys()),
            fill="toself",
            name=perfil_seleccionado,
            line_color=COLOR_GOLD
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Configuración pedagógica del perfil",
        paper_bgcolor=COLOR_LIGHT,
        font_color=COLOR_NAVY
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La visualización representa valores medios del perfil educativo seleccionado. "
        "No describe casos individuales ni emite juicios diagnósticos."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # Orientación pedagógica asociada al perfil
    # ---------------------------------------------------------
    st.subheader("Orientación pedagógica asociada al perfil")

    st.markdown(MENSAJES_PERFIL.get(perfil_seleccionado, ""))

    st.info(
    "La orientación describe **condiciones del entorno educativo** y posibles "
    "focos de observación. No constituye evaluación, diagnóstico ni clasificación individual."
    )

# =============================================================
# PESTAÑA 5 — METODOLOGÍA
# =============================================================

with tab_metodo:
    st.header("Metodología")

    st.markdown(
        """
        ### Enfoque pedagógico

        Esta herramienta se fundamenta en los principios de la **pedagogía Montessori**
        tal como son definidos por la *Asociación Montessori Internacional (AMI)*,
        donde la observación científica del niño precede a cualquier intervención.

        En este marco, el objetivo no es predecir conductas ni clasificar,
        sino **comprender patrones de relación entre el entorno, la autonomía y el bienestar**
        para favorecer una adaptación consciente del ambiente educativo.
        """
    )

    st.markdown(
        """
        ### Enfoque metodológico y técnico

        - Construcción de **índices pedagógicos** a partir de variables observables
        - Uso de **clustering no supervisado (K-Means)** para identificar patrones generales
        - Ausencia deliberada de modelos predictivos supervisados
        - Prioridad en la **interpretabilidad** sobre la precisión predictiva
        """
    )

    st.markdown(
        """
        ### Decisiones clave del diseño

        **Por qué no se utiliza un modelo predictivo supervisado**

        En coherencia con Montessori, no se dispone de un *ground truth* clínico
        ni se busca predecir resultados individuales.
        Utilizar modelos supervisados en este contexto podría inducir
        a interpretaciones deterministas o diagnósticas,
        contrarias al enfoque pedagógico de respeto al desarrollo.

        **Por qué se utilizan índices pedagógicos**

        Los índices permiten sintetizar observaciones complejas
        sin reducir al niño a una etiqueta,
        favoreciendo una lectura integrada y reflexiva del proceso educativo.
        """
    )

    st.info(
        """
        **Nota ética y pedagógica**

        Esta aplicación no emite diagnósticos, evaluaciones ni recomendaciones prescriptivas.
        Su función es **acompañar la observación pedagógica**
        y apoyar la reflexión del adulto responsable del entorno educativo.
        """
    )

# --- Pipeline ---
    st.subheader("Pipeline de datos")
    st.markdown("""
    1. **Carga y limpieza**: `StudentPerformanceFactors.csv` (6,607 registros). Nulos en `Teacher_Quality` (78) y `Parental_Education_Level` (90) imputados con la moda.
    2. **Mapeo de variables categóricas**: Low/Medium/High a 1/2/3, Yes/No a 1/0, Peer_Influence a 1/0/−1.
    3. **Normalización**: MinMaxScaler aplicado a componentes individuales antes de combinar en índices.
    4. **Construcción de índices**: ISEE, IAA, IBE, indice de observación educativa, calculados como combinaciones lineales ponderadas.
    6. **KMeans (k=4)**: Clustering sobre [ISEE, IAA, IBE] estandarizados con StandardScaler.
    """)
    st.divider()


# --- Citas Montessori ---
    st.subheader("Fundamentos pedagógicos")
    st.markdown("""
    > *"El niño no es un vaso que se llena, sino una fuente que se deja brotar."*
    > — Maria Montessori

    > *"Cuando el ambiente es adecuado, el niño trabaja y se construye a sí mismo."*
    > — *La mente absorbente del niño*

    > *"La ayuda innecesaria es un obstáculo para el desarrollo."*
    > — *El niño*

    > *"El desarrollo necesita tiempo y condiciones favorables."*
    > — *El niño en familia*

    > *"Sin equilibrio físico y emocional, el trabajo profundo no puede sostenerse."*
    > — *El niño*
    """)

    st.divider()

    # --- Contacto ---
    st.subheader("Créditos y contacto")
    st.markdown("""
    **Proyecto GURISES** — Un Caracol Montessori

    Herramienta de lectura pedagógica orientativa, desarrollada con fines educativos.
    No sustituye la observación profesional ni la evaluación clínica.

    Inspirada en la pedagogía Montessori y la visión de la Asociación Montessori Internacional (AMI).
    """)
