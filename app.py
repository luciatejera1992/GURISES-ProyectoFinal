import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CONSTANTES
# =========================================================
THRESHOLD_PEDAGOGICO = 0.65
THRESHOLD_CONDICION_BASE = 0.25
SENSIBILIDAD_IED = 0.3

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="GURISES — Un Caracol Montessori",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>

/* ================= BASE ================= */
.stApp {
    background-color: #ffffff;
}

/* ================= TIPOGRAFÍA ================= */
[data-testid="stTitle"] h1 {
    color: #4763a2 !important;
    font-weight: 700 !important;
    margin-bottom: 0.2em;
}

[data-testid="stHeader"] h2 {
    color: #4763a2 !important;
    font-weight: 700 !important;
    margin-top: 1.6em;
}

[data-testid="stSubheader"] h3 {
    color: #c48a0e !important;
    font-weight: 600 !important;
    margin-bottom: 0.6em;
}

p, li, label, span {
    font-size: 16px;
    line-height: 1.6;
}

/* ================= CAJAS MONTESSORI ================= */
.montessori-box {
    background-color: #f9fafc;
    border-left: 6px solid #c48a0e;
    padding: 1.6em;
    border-radius: 14px;
    margin-bottom: 1.6em;
}

/* ================= TARJETAS MÉTRICAS ================= */
.metric-card {
    background-color: #f9fafc;
    border-radius: 14px;
    padding: 1.4em;
    text-align: center;
    border: 1px solid #e0e4ed;
}

.metric-card h2 {
    color: #4763a2 !important;
    font-size: 2.2em;
    margin: 0;
    font-weight: 700;
}

.metric-card p {
    color: #555;
    font-size: 0.95em;
    margin: 0.3em 0 0 0;
}

/* ================= INPUTS ================= */
.montessori-box input,
.montessori-box select,
.montessori-box textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px !important;
}

.montessori-box [data-testid="stNumberInput"],
.montessori-box [data-testid="stSelectbox"],
.montessori-box [data-testid="stSlider"],
.montessori-box [data-testid="stTextInput"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 0.4em;
}

/* ================= BOTÓN ================= */
.stButton > button {
    background-color: #4763a2 !important;
    color: white !important;
    border-radius: 14px;
    padding: 0.7em 1.8em;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #36508c !important;
}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# RUTAS
# =========================================================
BASE_DIR = Path(__file__).parent

# =========================================================
# PALETA DE COLORES
# =========================================================
COLOR_NAVY = "#4763a2"
COLOR_GOLD = "#c48a0e"
COLOR_LIGHT = "#f9fafc"
COLOR_WHITE = "#ffffff"
PLOTLY_COLORS = ["#4763a2", "#c48a0e", "#6ba368", "#d4615e"]


# =========================================================
# CARGA DE MODELOS (con caché)
# =========================================================
@st.cache_resource
def cargar_modelos():
    modelos_dir = BASE_DIR / "models"
    with open(modelos_dir / "modelo_ml.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open(modelos_dir / "kmeans.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open(modelos_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return modelo, kmeans_model, scaler


# =========================================================
# CARGA DE DATOS (con caché)
# =========================================================
@st.cache_data
def cargar_datos_rendimiento():
    return pd.read_csv(BASE_DIR / "data" / "StudentPerformanceFactors.csv")


@st.cache_data
def cargar_datos_pantalla():
    return pd.read_csv(BASE_DIR / "data" / "screen_time.csv")


@st.cache_data
def cargar_ied_por_grupo():
    return pd.read_csv(BASE_DIR / "data" / "ied_by_age_group.csv")


@st.cache_data
def computar_indices(df_raw):
    """Reproduce el cálculo de ISEE, IAA, IBE desde el CSV crudo."""
    df = df_raw.copy()

    # --- Limpieza de nulos ---
    for col in ["Teacher_Quality", "Parental_Education_Level"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- ISEE ---
    level_map = {"Low": 1, "Medium": 2, "High": 3}
    for col in ["Parental_Involvement", "Access_to_Resources", "Teacher_Quality"]:
        df[col] = df[col].map(level_map)

    isee_raw = (
        0.4 * df["Parental_Involvement"]
        + 0.35 * df["Access_to_Resources"]
        + 0.25 * df["Teacher_Quality"]
    )
    df["ISEE"] = MinMaxScaler().fit_transform(isee_raw.values.reshape(-1, 1)).ravel()

    # --- IAA ---
    motivation_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Motivation_Level"] = df["Motivation_Level"].map(motivation_map)

    norm_cols = ["Hours_Studied", "Attendance", "Motivation_Level", "Tutoring_Sessions"]
    scaler_iaa = MinMaxScaler()
    df[norm_cols] = scaler_iaa.fit_transform(df[norm_cols])

    iaa_raw = (
        0.30 * df["Hours_Studied"]
        + 0.30 * df["Attendance"]
        + 0.25 * df["Motivation_Level"]
        - 0.15 * df["Tutoring_Sessions"]
    )
    df["IAA"] = MinMaxScaler().fit_transform(iaa_raw.values.reshape(-1, 1)).ravel()

    # --- IBE ---
    peer_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["Peer_Influence_num"] = df["Peer_Influence"].map(peer_map)
    df["Peer_Influence_norm"] = (df["Peer_Influence_num"] - (-1)) / (1 - (-1))

    ibe_scaler = MinMaxScaler()
    df[["Sleep_Hours_norm", "Physical_Activity_norm"]] = ibe_scaler.fit_transform(
        df[["Sleep_Hours", "Physical_Activity"]]
    )

    ibe_raw = (
        0.30 * df["Sleep_Hours_norm"]
        + 0.25 * df["Physical_Activity_norm"]
        + 0.20 * df["Peer_Influence_norm"]
        + 0.15 * df["Motivation_Level"]
        + 0.10 * df["Attendance"]
    )
    df["IBE"] = MinMaxScaler().fit_transform(ibe_raw.values.reshape(-1, 1)).ravel()

    # --- Target: Desajuste ---
    df["Score_Desajuste"] = 0
    df.loc[(df["ISEE"] > 0.65) & (df["IAA"] < 0.35), "Score_Desajuste"] += 2
    df.loc[(df["ISEE"] < 0.35) & (df["IAA"] > 0.65), "Score_Desajuste"] += 2
    df.loc[(df["ISEE"] > 0.6) & (df["IAA"] < 0.45), "Score_Desajuste"] += 1
    df.loc[(df["ISEE"] < 0.4) & (df["IAA"] > 0.55), "Score_Desajuste"] += 1
    df.loc[(df["IBE"] < 0.4) & (df["IAA"] > 0.5), "Score_Desajuste"] += 1
    df["Desajuste"] = np.where(df["Score_Desajuste"] >= 1, 1, 0)

    return df


@st.cache_data
def obtener_metricas_modelo(df):
    """Entrena el modelo en vivo para obtener métricas reproducibles."""
    X = df[["ISEE", "IAA", "IBE"]]
    y = df["Desajuste"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report["accuracy"]

    # Silhouette del KMeans existente
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X)
    from sklearn.cluster import KMeans as _KM
    km = _KM(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    centroids_df = pd.DataFrame(
        km.cluster_centers_,
        columns=["ISEE", "IAA", "IBE"],
    )
    centroids_df.index.name = "Cluster"

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    return {
        "report": report,
        "accuracy": accuracy,
        "silhouette": sil,
        "y_proba": y_proba,
        "y_test": y_test,
        "centroids": centroids_df,
        "cluster_counts": cluster_counts,
        "labels": labels,
    }


# =========================================================
# MENSAJES POR PERFIL EDUCATIVO
# =========================================================
MENSAJES_PERFIL = {
    0: """
    **Perfil educativo equilibrado**

    El entorno, la autonomía y el bienestar están en armonía, favoreciendo el desarrollo natural.
    Recomendación: Mantener la coherencia del ambiente y observar sin intervenir innecesariamente.

    *Fundamento Montessori*: "Cuando el ambiente es adecuado, el niño trabaja y se construye a sí mismo."
    (*La mente absorbente del niño*)
    """,
    1: """
    **Entorno favorable, autonomía en construcción**

    El entorno es adecuado, pero la autonomía está en desarrollo. Reflexionar sobre:
    - Grado de ayuda ofrecida.
    - Oportunidades reales de elección.
    - Tiempo permitido para el error y la repetición.

    *Fundamento Montessori*: "La ayuda innecesaria es un obstáculo para el desarrollo."
    (*El niño*)
    """,
    2: """
    **Autonomía alta, entorno exigente**

    Aunque la autonomía es alta, el entorno puede ser demasiado exigente. Recomendación:
    - Simplificar el ambiente.
    - Reducir estímulos y expectativas externas.

    *Fundamento Montessori*: "El desarrollo necesita tiempo y condiciones favorables."
    (*El niño en familia*)
    """,
    3: """
    **Bienestar comprometido**

    El bienestar está afectado, limitando el aprendizaje. Prioridad:
    - Restablecer calma y equilibrio emocional.
    - Reducir demandas innecesarias.

    *Fundamento Montessori*: "Sin equilibrio físico y emocional, el trabajo profundo no puede sostenerse."
    (*El niño*)
    """,
}

MENSAJES_DESAJUSTE = {
    0: """
    **Señal leve de desajuste**

    Puede haber una incoherencia puntual entre el entorno y el desarrollo actual. Recomendación:
    - Observar con atención.
    - Ajustar pequeñas variables del ambiente si es necesario.
    """,
    1: """
    **Exceso de ayuda o estructuración**

    Revisar si el adulto está anticipándose a procesos que el niño podría asumir. Recomendación:
    - Retirar apoyos innecesarios progresivamente.
    """,
    2: """
    **Exigencias ambientales elevadas**

    El entorno puede estar sobrestimulado o con expectativas altas. Recomendación:
    - Reducir presión externa.
    - Ofrecer espacios de pausa y reflexión.
    """,
    3: """
    **Bienestar comprometido**

    El bienestar está afectado. Prioridad:
    - Restablecer calma y equilibrio emocional antes de nuevas intervenciones.
    """,
}


# =========================================================
# FUNCIÓN DE AJUSTE POR ESCENARIO DIGITAL (IED)
# =========================================================
def ajustar_riesgo_por_ied(probabilidad_base, ied):
    ajuste = 1 + SENSIBILIDAD_IED * (0.5 - ied)
    return min(max(probabilidad_base * ajuste, 0), 1)


# =========================================================
# CARGA INICIAL
# =========================================================
try:
    model, kmeans, scaler_cluster = cargar_modelos()
except FileNotFoundError as e:
    st.error(f"No se encontró un archivo de modelo: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

logo = Image.open(BASE_DIR / "assets" / "logo.png")

# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(logo, width=100)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0;color:#4763a2;'>GURISES</h1>"
        "<p style='margin-top:0;color:#555;font-size:1.1em;'>"
        "Un Caracol Montessori — Herramienta de lectura pedagógica</p>",
        unsafe_allow_html=True,
    )

# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================
tab_inicio, tab_eval, tab_datos, tab_modelo, tab_metodo = st.tabs([
    "Inicio",
    "Evaluación",
    "Exploración de datos",
    "Modelo y clustering",
    "Metodología",
])


# =============================================================
# PESTAÑA 1 — INICIO
# =============================================================
with tab_inicio:
    st.markdown("")

    st.markdown("""
    <div class="montessori-box">
    <p>
    Esta herramienta se inspira en la <strong>pedagogía Montessori</strong> y en la visión
    del desarrollo infantil promovida por la <em>Asociación Montessori Internacional (AMI)</em>.
    </p>
    <p>
    No evalúa, no clasifica ni etiqueta. Propone una <strong>lectura orientativa</strong>
    del equilibrio entre el entorno, la autonomía y el bienestar.
    </p>
    <p>
    Invita a <strong>observar el ambiente</strong> y reflexionar
    sobre cómo puede ajustarse para acompañar mejor
    el desarrollo natural de cada criatura.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Tarjetas métricas ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h2>6,607</h2>'
            "<p>Registros analizados</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h2>4</h2>'
            "<p>Índices pedagógicos</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h2>4</h2>'
            "<p>Perfiles educativos</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Los 4 índices pedagógicos")

    idx_col1, idx_col2 = st.columns(2)
    with idx_col1:
        st.markdown("""
        **ISEE — Entorno preparado**

        Mide la calidad del ambiente educativo: orden, recursos, apoyo parental
        y calidad docente. Un entorno preparado facilita la autonomía y la concentración.

        **IAA — Autonomía y autodisciplina**

        Evalúa la capacidad del niño para iniciar y sostener actividades por cuenta propia,
        mantener el interés y depender menos de estímulos externos.
        """)
    with idx_col2:
        st.markdown("""
        **IBE — Bienestar y equilibrio**

        Refleja el estado físico, emocional y social del niño. El bienestar es condición
        indispensable para el aprendizaje profundo.

        **IED — Entorno digital**

        Considera cómo el uso de pantallas influye en el desarrollo.
        Se utiliza como factor de ajuste en la lectura pedagógica.
        """)

    st.divider()
    st.markdown("""
    ### Cómo funciona

    1. **Evaluación**: Ingresa las observaciones sobre el niño en 4 dimensiones.
    2. **Perfil educativo**: Un modelo de clustering (KMeans) identifica el perfil.
    3. **Señal de desajuste**: Un modelo de clasificación (Regresión Logística) detecta incoherencias.
    4. **Ajuste digital**: El entorno digital modula la lectura final.
    """)


# =============================================================
# PESTAÑA 2 — EVALUACIÓN
# =============================================================
with tab_eval:
    st.markdown("")

    with st.form("formulario_evaluacion"):

        # ----- 1. ENTORNO PREPARADO (ISEE) -----
        st.subheader("1. Entorno preparado (ISEE)")
        st.markdown("""
        <div class="montessori-box">
        <p>
        En la pedagogía Montessori, el <strong>entorno</strong> es considerado
        un elemento educativo fundamental.
        </p>
        <p>Un entorno preparado es aquel que:</p>
        <ul>
        <li>ofrece orden, claridad y previsibilidad,</li>
        <li>permite al niño actuar con independencia real,</li>
        <li>acompaña sin interferir ni sobreproteger.</li>
        </ul>
        <p>
        Este indicador describe <strong>cómo el ambiente actual puede facilitar
        o dificultar la autonomía, la concentración y el bienestar</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)

        isee_options = {
            0.0: "Entorno poco preparado: desorden, exceso de estímulos o falta de apoyo adecuado.",
            0.25: "Entorno con muchas dificultades para sostener la concentración y la autonomía.",
            0.5: "Entorno medianamente preparado, con aspectos positivos y otros a mejorar.",
            0.75: "Entorno mayormente preparado, con buen acompañamiento y estructura.",
            1.0: "Entorno cuidadosamente preparado que favorece autonomía, calma y aprendizaje.",
        }
        isee = st.selectbox(
            "¿Cómo describirías el entorno del niño?",
            options=list(isee_options.keys()),
            format_func=lambda x: isee_options[x],
            help="Selecciona la opción que mejor represente el entorno habitual.",
            key="isee_selectbox",
        )

        st.divider()

        # ----- 2. AUTONOMÍA Y AUTODISCIPLINA (IAA) -----
        st.subheader("2. Autonomía y autodisciplina (IAA)")
        st.markdown("""
        <div class="montessori-box">
        <p>
        La autonomía, desde Montessori, no significa "hacer todo solo",
        sino desarrollar la capacidad de iniciar, sostener y regular la propia actividad.
        </p>
        <p>Este indicador refleja el grado en que el niño o adolescente:</p>
        <ul>
        <li>actúa por iniciativa propia,</li>
        <li>mantiene el interés en una tarea,</li>
        <li>depende o no de estímulos externos constantes.</li>
        </ul>
        <p>
        La autodisciplina es entendida aquí como una construcción interna,
        no como obediencia externa.
        </p>
        </div>
        """, unsafe_allow_html=True)

        iaa_options = {
            0.0: "Alta dependencia del adulto para iniciar y sostener actividades.",
            0.25: "Autonomía muy incipiente; necesita guía constante.",
            0.5: "Muestra autonomía en algunos momentos, pero no de forma estable.",
            0.75: "Buen nivel de iniciativa y autorregulación.",
            1.0: "Autonomía consolidada y autodisciplina interna.",
        }
        iaa = st.selectbox(
            "¿Cómo describirías el nivel de autonomía del niño?",
            options=list(iaa_options.keys()),
            format_func=lambda x: iaa_options[x],
            help="Considera la capacidad del niño para iniciar y sostener actividades sin depender excesivamente del adulto.",
        )

        st.divider()

        # ----- 3. BIENESTAR Y EQUILIBRIO (IBE) -----
        st.subheader("3. Bienestar y equilibrio (IBE)")
        st.markdown("""
        <div class="montessori-box">
        <p>
        El bienestar es una condición indispensable para el aprendizaje profundo.
        </p>
        <p>
        Desde la pedagogía Montessori, el niño solo puede concentrarse y aprender
        cuando existe un equilibrio razonable entre su estado físico,
        emocional y social.
        </p>
        <p>
        Este indicador no mide estados clínicos,
        sino la disponibilidad del niño para aprender y desarrollarse
        en el contexto actual.
        </p>
        </div>
        """, unsafe_allow_html=True)

        ibe_options = {
            0.0: "Malestar importante que interfiere con el aprendizaje.",
            0.25: "Frecuente desequilibrio físico o emocional.",
            0.5: "Bienestar intermedio, con altibajos.",
            0.75: "Buen equilibrio general.",
            1.0: "Estado de bienestar estable que favorece la concentración y el interés.",
        }
        ibe = st.selectbox(
            "¿Cómo describirías el bienestar general del niño?",
            options=list(ibe_options.keys()),
            format_func=lambda x: ibe_options[x],
            help="Considera el equilibrio físico, emocional y social del niño en su entorno actual.",
        )

        st.divider()

        # ----- 4. ENTORNO DIGITAL (IED) -----
        st.subheader("4. Entorno digital (IED)")
        st.markdown("""
        <div class="montessori-box">
        <p>
        Las pantallas forman parte del mundo actual, pero su uso debe estar equilibrado.
        </p>
        <p>
        Este indicador permite considerar cómo distintos entornos digitales
        pueden influir en el desarrollo.
        </p>
        </div>
        """, unsafe_allow_html=True)

        ied_options = {
            0.0: "Uso digital muy desequilibrado, con impacto negativo.",
            0.25: "Uso frecuente con poco acompañamiento.",
            0.5: "Uso moderado, con equilibrio variable.",
            0.75: "Uso mayormente equilibrado y acompañado.",
            1.0: "Uso digital consciente, educativo y bien integrado.",
        }
        ied = st.selectbox(
            "¿Cómo describirías el entorno digital?",
            options=list(ied_options.keys()),
            format_func=lambda x: ied_options[x],
            help="Considera la frecuencia, el tipo de uso y el acompañamiento adulto.",
        )

        st.divider()
        enviado = st.form_submit_button("Observar lectura pedagógica")

    # --- RESULTADOS ---
    if enviado:
        input_df = pd.DataFrame(
            [[isee, iaa, ibe]], columns=["ISEE", "IAA", "IBE"]
        )
        condicion_base_ok = not (
            (isee <= THRESHOLD_CONDICION_BASE)
            and (iaa <= THRESHOLD_CONDICION_BASE)
            and (ibe <= THRESHOLD_CONDICION_BASE)
        )

        input_cluster_scaled = scaler_cluster.transform(input_df)
        perfil_predicho = kmeans.predict(input_cluster_scaled)[0]

        riesgo_base = model.predict_proba(input_df)[0][1]
        riesgo_ajustado = ajustar_riesgo_por_ied(riesgo_base, ied)

        st.divider()
        st.subheader("Lectura pedagógica del desarrollo")

        if not condicion_base_ok:
            st.warning("""
            **Condiciones básicas del desarrollo comprometidas**

            Los niveles actuales de entorno, autonomía y bienestar
            se encuentran por debajo de lo necesario para sostener
            un proceso de desarrollo pleno.

            Desde la pedagogía Montessori, la prioridad es
            restablecer condiciones básicas del ambiente
            antes de interpretar perfiles educativos o señales de desajuste.
            """)
        else:
            st.markdown(MENSAJES_PERFIL[perfil_predicho])

            if riesgo_ajustado >= THRESHOLD_PEDAGOGICO:
                st.warning(MENSAJES_DESAJUSTE[perfil_predicho])
            else:
                st.success("""
                **Sin señales significativas de desajuste**

                El modelo no detecta incoherencias relevantes entre
                el entorno, la autonomía y el bienestar en este momento.

                Recomendación: continuar observando y acompañando
                el proceso de desarrollo con la misma coherencia.
                """)

        # --- Radar del perfil ---
        st.divider()
        st.subheader("Perfil del niño evaluado")
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[isee, iaa, ibe, isee],
            theta=["ISEE<br>Entorno", "IAA<br>Autonomía", "IBE<br>Bienestar", "ISEE<br>Entorno"],
            fill="toself",
            fillcolor="rgba(71, 99, 162, 0.25)",
            line=dict(color=COLOR_NAVY, width=2),
            name="Niño evaluado",
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1]),
            ),
            showlegend=False,
            height=380,
            margin=dict(t=40, b=40, l=60, r=60),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        # --- Entorno digital ---
        st.divider()
        st.subheader("Influencia del entorno digital")

        if ied <= 0.25:
            st.warning("""
            **Entorno digital con impacto significativo**

            El uso actual de pantallas puede estar interfiriendo
            en la concentración y el equilibrio del niño.
            Considerar reducir exposición y aumentar acompañamiento.
            """)
        elif ied <= 0.5:
            st.info("""
            **Entorno digital con margen de mejora**

            El uso de pantallas es moderado pero podría beneficiarse
            de mayor intencionalidad y acompañamiento adulto.
            """)
        else:
            st.success("""
            **Entorno digital equilibrado**

            El uso de pantallas parece estar bien integrado,
            con propósito educativo y acompañamiento adecuado.
            """)

        # --- Indicadores técnicos ---
        st.divider()
        st.caption("Indicador técnico de referencia")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad base", f"{riesgo_base:.0%}")
        with col2:
            st.metric("Ajustada (entorno digital)", f"{riesgo_ajustado:.0%}")

        st.caption("""
        La lectura pedagógica se basa en un umbral cuidadosamente definido.
        El modelo solo señala desajuste cuando la probabilidad supera el 65%,
        para evitar alertas innecesarias y respetar los ritmos naturales del desarrollo.
        """)

        # --- Reflexión final ---
        st.divider()
        st.markdown("""
        ### Reflexión final

        En la pedagogía Montessori, el desarrollo no se mide por resultados
        inmediatos, sino por la **coherencia entre el niño y su entorno**.

        Esta herramienta no ofrece diagnósticos ni etiquetas.
        Ofrece **información para observar, comprender y acompañar mejor**.

        Cuando surge una señal de desajuste, la pregunta no es:
        *¿qué le pasa al niño?*,
        sino:
        *¿qué necesita el ambiente para acompañarlo mejor?*
        """)


# =============================================================
# PESTAÑA 3 — EXPLORACIÓN DE DATOS
# =============================================================
with tab_datos:
    st.markdown("")

    sub_rend, sub_screen = st.tabs([
        "Rendimiento estudiantil",
        "Tiempo de pantalla",
    ])

    # --- Subpestaña: Rendimiento estudiantil ---
    with sub_rend:
        df_raw = cargar_datos_rendimiento()
        df_idx = computar_indices(df_raw)

        st.subheader("Distribución de los índices pedagógicos")
        idx_to_show = st.selectbox(
            "Selecciona un índice",
            ["ISEE", "IAA", "IBE"],
            key="hist_idx_select",
        )
        nombres_idx = {
            "ISEE": "Entorno preparado (ISEE)",
            "IAA": "Autonomía y autodisciplina (IAA)",
            "IBE": "Bienestar y equilibrio (IBE)",
        }
        fig_hist = px.histogram(
            df_idx,
            x=idx_to_show,
            nbins=40,
            title=nombres_idx[idx_to_show],
            labels={idx_to_show: idx_to_show},
            color_discrete_sequence=[COLOR_NAVY],
        )
        fig_hist.update_layout(
            yaxis_title="Frecuencia",
            bargap=0.05,
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # --- Heatmap de correlación ---
        st.subheader("Correlación entre índices")
        corr = df_idx[["ISEE", "IAA", "IBE"]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=["#ffffff", COLOR_NAVY],
            zmin=-1,
            zmax=1,
            title="Matriz de correlación (ISEE, IAA, IBE)",
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()

        # --- Boxplots con datos originales ---
        st.subheader("Variables del dataset original")

        df_box = cargar_datos_rendimiento()
        # Limpieza para Teacher_Quality y School_Type
        for col in ["Teacher_Quality", "Parental_Education_Level"]:
            if col in df_box.columns and df_box[col].isnull().any():
                df_box[col] = df_box[col].fillna(df_box[col].mode()[0])

        box_col1, box_col2 = st.columns(2)

        with box_col1:
            fig_tq = px.box(
                df_box,
                x="Teacher_Quality",
                y="Exam_Score",
                color="Teacher_Quality",
                color_discrete_sequence=PLOTLY_COLORS,
                title="Exam Score por Teacher Quality",
                category_orders={"Teacher_Quality": ["Low", "Medium", "High"]},
            )
            fig_tq.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_tq, use_container_width=True)

        with box_col2:
            fig_st = px.box(
                df_box,
                x="School_Type",
                y="Exam_Score",
                color="School_Type",
                color_discrete_sequence=PLOTLY_COLORS,
                title="Exam Score por School Type",
            )
            fig_st.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_st, use_container_width=True)

        st.divider()

        # --- Scatter Hours_Studied vs Exam_Score ---
        st.subheader("Hours Studied vs Exam Score")
        fig_scatter = px.scatter(
            df_box,
            x="Hours_Studied",
            y="Exam_Score",
            opacity=0.3,
            color_discrete_sequence=[COLOR_NAVY],
            title="Relación entre horas de estudio y calificación",
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Subpestaña: Tiempo de pantalla ---
    with sub_screen:
        df_screen = cargar_datos_pantalla()

        st.subheader("IED por grupo etario")
        df_ied = cargar_ied_por_grupo()
        fig_ied = px.bar(
            df_ied,
            x="Age_Group",
            y="IED",
            color="Age_Group",
            color_discrete_sequence=[COLOR_NAVY, COLOR_GOLD],
            title="Índice de Entorno Digital por grupo etario",
            text_auto=".2f",
        )
        fig_ied.update_layout(showlegend=False, height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig_ied, use_container_width=True)

        st.divider()

        # --- Boxplot por tipo de día ---
        st.subheader("Tiempo de pantalla por tipo de día")
        fig_day = px.box(
            df_screen,
            x="Day Type",
            y="Average Screen Time (hours)",
            color="Day Type",
            color_discrete_sequence=[COLOR_NAVY, COLOR_GOLD],
            title="Distribución del tiempo de pantalla: Weekday vs Weekend",
        )
        fig_day.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_day, use_container_width=True)

        st.divider()

        # --- Distribución por tipo de pantalla ---
        st.subheader("Tiempo de pantalla por tipo de uso")
        fig_type = px.box(
            df_screen,
            x="Screen Time Type",
            y="Average Screen Time (hours)",
            color="Screen Time Type",
            color_discrete_sequence=PLOTLY_COLORS[:3],
            title="Educational vs Recreational vs Total",
        )
        fig_type.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_type, use_container_width=True)


# =============================================================
# PESTAÑA 4 — MODELO Y CLUSTERING
# =============================================================
with tab_modelo:
    st.markdown("")

    df_raw_m = cargar_datos_rendimiento()
    df_m = computar_indices(df_raw_m)
    metricas = obtener_metricas_modelo(df_m)

    # --- Métricas principales ---
    st.subheader("Métricas del modelo")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Accuracy", f"{metricas['accuracy']:.1%}")
    with m_col2:
        st.metric("Silhouette Score (k=4)", f"{metricas['silhouette']:.3f}")
    with m_col3:
        f1_macro = metricas["report"]["macro avg"]["f1-score"]
        st.metric("F1 (macro avg)", f"{f1_macro:.2f}")

    st.divider()

    # --- Classification Report ---
    st.subheader("Classification Report")
    report = metricas["report"]
    report_rows = []
    for label in ["0", "1"]:
        if label in report:
            report_rows.append({
                "Clase": "Sin desajuste" if label == "0" else "Con desajuste",
                "Precision": f"{report[label]['precision']:.2f}",
                "Recall": f"{report[label]['recall']:.2f}",
                "F1-Score": f"{report[label]['f1-score']:.2f}",
                "Support": int(report[label]["support"]),
            })
    report_rows.append({
        "Clase": "**Macro avg**",
        "Precision": f"{report['macro avg']['precision']:.2f}",
        "Recall": f"{report['macro avg']['recall']:.2f}",
        "F1-Score": f"{report['macro avg']['f1-score']:.2f}",
        "Support": int(report['macro avg']['support']),
    })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    st.divider()

    # --- Distribución de perfiles (pie chart) ---
    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.subheader("Distribución de perfiles")
        counts = metricas["cluster_counts"]
        perfil_names = {
            0: "Equilibrado",
            1: "Autonomía en construcción",
            2: "Entorno exigente",
            3: "Bienestar comprometido",
        }
        fig_pie = px.pie(
            names=[perfil_names.get(i, f"Cluster {i}") for i in counts.index],
            values=counts.values,
            color_discrete_sequence=PLOTLY_COLORS,
            title="Distribución de perfiles educativos (KMeans k=4)",
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with model_col2:
        st.subheader("Centroides de los clusters")
        centroids = metricas["centroids"].copy()
        centroids.insert(0, "Perfil", [perfil_names.get(i, f"Cluster {i}") for i in centroids.index])
        for col in ["ISEE", "IAA", "IBE"]:
            centroids[col] = centroids[col].map("{:.3f}".format)
        st.dataframe(centroids, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="montessori-box" style="margin-top:1em;">
        <p><strong>Nota:</strong> Los centroides están en escala estandarizada (StandardScaler).
        Valores negativos indican "por debajo de la media" y positivos "por encima".</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Histograma de probabilidad de desajuste ---
    st.subheader("Distribución de la probabilidad de desajuste")
    fig_proba = px.histogram(
        x=metricas["y_proba"],
        nbins=40,
        color_discrete_sequence=[COLOR_NAVY],
        labels={"x": "Probabilidad predicha de desajuste"},
        title="Distribución de probabilidades en el conjunto de test",
    )
    fig_proba.add_vline(
        x=THRESHOLD_PEDAGOGICO,
        line_dash="dash",
        line_color=COLOR_GOLD,
        annotation_text=f"Umbral = {THRESHOLD_PEDAGOGICO}",
        annotation_position="top right",
    )
    fig_proba.update_layout(
        yaxis_title="Frecuencia",
        height=400,
        bargap=0.05,
    )
    st.plotly_chart(fig_proba, use_container_width=True)


# =============================================================
# PESTAÑA 5 — METODOLOGÍA
# =============================================================
with tab_metodo:
    st.markdown("")

    st.subheader("Construcción de los índices")

    st.markdown("""
    <div class="montessori-box">
    <h4 style="color:#4763a2;">ISEE — Índice Socioeducativo del Entorno</h4>
    <p><code>ISEE = MinMaxScaler(0.40 × Parental_Involvement + 0.35 × Access_to_Resources + 0.25 × Teacher_Quality)</code></p>
    <p>Variables originales mapeadas: Low=1, Medium=2, High=3. Resultado normalizado a [0, 1].</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="montessori-box">
    <h4 style="color:#4763a2;">IAA — Índice de Autonomía y Autodisciplina</h4>
    <p><code>IAA = MinMaxScaler(0.30 × Hours_Studied + 0.30 × Attendance + 0.25 × Motivation_Level − 0.15 × Tutoring_Sessions)</code></p>
    <p>Variables previamente normalizadas con MinMaxScaler. Tutoring_Sessions resta porque indica dependencia del adulto.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="montessori-box">
    <h4 style="color:#4763a2;">IBE — Índice de Bienestar y Equilibrio</h4>
    <p><code>IBE = MinMaxScaler(0.30 × Sleep_Hours_norm + 0.25 × Physical_Activity_norm + 0.20 × Peer_Influence_norm + 0.15 × Motivation_Level + 0.10 × Attendance)</code></p>
    <p>Sleep_Hours y Physical_Activity normalizados a [0, 1] antes de combinar. Peer_Influence: Positive=1, Neutral=0, Negative=−1, normalizado a [0, 1].</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="montessori-box">
    <h4 style="color:#4763a2;">IED — Índice de Entorno Digital</h4>
    <p>Calculado a partir del dataset <code>screen_time.csv</code> como proporción de tiempo educativo sobre tiempo total, agrupado por edad.</p>
    <p>En la app se utiliza como factor de ajuste: <code>riesgo_ajustado = riesgo_base × (1 + 0.3 × (0.5 − IED))</code></p>
    <p>Rango de ajuste: ±15% sobre la probabilidad base.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Pipeline ---
    st.subheader("Pipeline de datos")
    st.markdown("""
    1. **Carga y limpieza**: `StudentPerformanceFactors.csv` (6,607 registros). Nulos en `Teacher_Quality` (78) y `Parental_Education_Level` (90) imputados con la moda.
    2. **Mapeo de variables categóricas**: Low/Medium/High a 1/2/3, Yes/No a 1/0, Peer_Influence a 1/0/−1.
    3. **Normalización**: MinMaxScaler aplicado a componentes individuales antes de combinar en índices.
    4. **Construcción de índices**: ISEE, IAA, IBE calculados como combinaciones lineales ponderadas.
    5. **Variable objetivo**: `Desajuste` construido con reglas sobre ISEE, IAA, IBE (Score_Desajuste ≥ 1 → 1).
    6. **KMeans (k=4)**: Clustering sobre [ISEE, IAA, IBE] estandarizados con StandardScaler.
    7. **Regresión Logística**: Pipeline StandardScaler + LogisticRegression (class_weight="balanced", max_iter=1000), split 80/20 estratificado.
    8. **IED**: Proporción de screen time educativo desde `screen_time.csv`, agrupado por Child/Adolescent.
    """)

    st.divider()

    # --- Limitaciones ---
    st.subheader("Limitaciones conocidas")
    st.markdown("""
    **1. Variable objetivo circular.**
    `Desajuste` se construye con reglas manuales sobre ISEE, IAA, IBE — las mismas variables usadas como features.
    El modelo aproxima las reglas, no aprende de datos reales externos. Esto explica el accuracy de ~63%.

    **2. Data leakage en scalers.**
    Los MinMaxScaler se ajustan sobre el dataset completo antes del train/test split.

    **3. IED constante.**
    IED se asigna como promedio (~0.5) a las 6,607 filas porque no hay datos individuales de uso de pantalla.

    **4. Cluster labels hardcodeados.**
    Las etiquetas pedagógicas están asignadas a IDs de cluster fijos. Si se reentrena KMeans, los IDs pueden cambiar.

    **5. Sin cross-validation.**
    Solo se usa un split 80/20 sin validación cruzada ni comparación de modelos.
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
