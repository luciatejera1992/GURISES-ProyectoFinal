import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Un Caracol Montessori",
    layout="centered",
    initial_sidebar_state="collapsed"
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
    color: #f1a716 !important;
    font-weight: 600 !important;
    margin-bottom: 0.6em;
}

p, li, label, span {
    color: #000000 !important;
    font-size: 16px;
    line-height: 1.6;
}

/* ================= CAJAS MONTESSORI ================= */
.montessori-box {
    background-color: #f9fafc;
    border-left: 6px solid #f1a716;
    padding: 1.6em;
    border-radius: 14px;
    margin-bottom: 1.6em;
}

/* ================= INPUTS ================= */
.montessori-box input,
.montessori-box select,
.montessori-box textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px !important;
}

/* Contenedores Streamlit */
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

/* ================= SEPARADOR ================= */
hr {
    border: none;
    height: 1px;
    background-color: #e6e6e6;
    margin: 2em 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER – MARCA - LOGO
# =========================================================

from PIL import Image

logo = Image.open("logo.png")


# Logo
st.image(logo, width=140)

# Espacio suave
st.markdown("<br>", unsafe_allow_html=True)

# Nombre de la marca
st.markdown(
    "<h1 style='margin-bottom:0.2em;'>Un Caracol Montessori </h1>",
    unsafe_allow_html=True
)


st.markdown('</div>', unsafe_allow_html=True)
# Separador

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# CARGA DE MODELOS
# =========================================================
with open("modelo_ml.pkl", "rb") as f:
    model = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler_cluster = pickle.load(f)

THRESHOLD_PEDAGOGICO = 0.65

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
    """
}

# =========================================================
# MENSAJES DE DESAJUSTE PEDAGÓGICO
# =========================================================
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
    """
}

# =========================================================
# FUNCIÓN DE AJUSTE POR ESCENARIO DIGITAL (IED)
# =========================================================
def ajustar_riesgo_por_ied(probabilidad_base, ied, sensibilidad=0.3):
    ajuste = 1 + sensibilidad * (0.5 - ied)
    return min(max(probabilidad_base * ajuste, 0), 1)

# =========================================================
# TÍTULO Y CONTEXTO
# =========================================================
st.title("GURISES")


st.markdown("""

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

""", unsafe_allow_html=True)

st.divider()

# =========================================================
# 1. ENTORNO PREPARADO (ISEE)
# =========================================================

st.subheader("1. Entorno preparado (ISEE)")

st.markdown("""
<div class="montessori-box">
<p>
En la pedagogía Montessori, el <strong>entorno</strong> es considerado
un elemento educativo fundamental.
</p>

<p>
Un entorno preparado es aquel que:
</p>

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
    1.0: "Entorno cuidadosamente preparado que favorece autonomía, calma y aprendizaje."
}




isee = st.selectbox(
        "¿Cómo describirías el entorno del niño?",
        options=list(isee_options.keys()),
        format_func=lambda x: isee_options[x],
        help="Selecciona la opción que mejor represente el entorno habitual.",
        key="isee_selectbox"
    )



st.divider()

# =========================================================
# SECCIÓN IAA — AUTONOMÍA Y AUTODISCIPLINA
# =========================================================
st.subheader("2. Autonomía y autodisciplina (IAA)")

st.markdown(""" <div class="montessori-box">
<p>
La autonomía, desde Montessori, no significa “hacer todo solo”,
sino desarrollar la capacidad de iniciar, sostener y regular la propia actividad.
</p>

<p> Este indicador refleja el grado en que el niño o adolescente:</p>

<ul>
<li> actúa por iniciativa propia,</li>
<li> mantiene el interés en una tarea,</li>
<li> depende o no de estímulos externos constantes.</li>
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
    1.0: "Autonomía consolidada y autodisciplina interna."
}

iaa_label = st.selectbox(
    "¿Cómo describirías el nivel de autonomía del niño?",
    options=list(iaa_options.keys()),
    format_func=lambda x: iaa_options[x],
    help="Considera la capacidad del niño para iniciar y sostener actividades sin depender excesivamente del adulto."
)

iaa = iaa_label

st.divider()

# =========================================================
# SECCIÓN IBE — BIENESTAR Y EQUILIBRIO
# =========================================================
st.subheader("3. Bienestar y equilibrio (IBE)")

st.markdown(""" <div class="montessori-box">
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
sino **la disponibilidad del niño para aprender y desarrollarse
en el contexto actual**.
</p>
</div>
""", unsafe_allow_html=True)


ibe_options = {
    0.0: "Malestar importante que interfiere con el aprendizaje.",
    0.25: "Frecuente desequilibrio físico o emocional.",
    0.5: "Bienestar intermedio, con altibajos.",
    0.75: "Buen equilibrio general.",
    1.0: "Estado de bienestar estable que favorece la concentración y el interés."
}

ibe_label = st.selectbox(
    "¿Cómo describirías el bienestar general del niño?",
    options=list(ibe_options.keys()),
    format_func=lambda x: ibe_options[x],
    help="Considera el equilibrio físico, emocional y social del niño en su entorno actual."
)

ibe = ibe_label



# =========================================================
# PREPARACIÓN DE DATOS
# =========================================================
input_df = pd.DataFrame(
    [[isee, iaa, ibe]],
    columns=["ISEE", "IAA", "IBE"]
)

CONDICION_BASE_OK = not (
    (isee <= 0.35) and
    (iaa <= 0.35) and
    (ibe <= 0.35)
)

# =========================================================
# PERFIL EDUCATIVO (CLUSTERING)
# =========================================================


# Escalar los índices con el mismo scaler usado en entrenamiento
input_cluster_scaled = scaler_cluster.transform(input_df)

# Predecir perfil educativo
perfil_predicho = kmeans.predict(input_cluster_scaled)[0]


PERFILES = {
    0: "Perfil educativo equilibrado",
    1: "Perfil con entorno favorable y autonomía en construcción",
    2: "Perfil con autonomía alta y entorno exigente",
    3: "Perfil con bienestar comprometido"
}




# =========================================================
# RIESGO EDUCATIVO BASE
# =========================================================
riesgo_base = model.predict_proba(input_df)[0][1]


st.divider()
st.subheader("Lectura pedagógica del desarrollo")

# CASO 1 — CONDICIÓN BASE COMPROMETIDA
if not CONDICION_BASE_OK:
    st.warning("""
    **Condiciones básicas del desarrollo comprometidas**

    Los niveles actuales de entorno, autonomía y bienestar
    se encuentran por debajo de lo necesario para sostener
    un proceso de desarrollo pleno.

    Desde la pedagogía Montessori, la prioridad es
    restablecer condiciones básicas del ambiente
    antes de interpretar perfiles educativos o señales de desajuste.
    """)

# CASO 2 — HAY CONDICIÓN BASE
else:
    # PERFIL EDUCATIVO
    st.markdown(MENSAJES_PERFIL[perfil_predicho])


    # DESAJUSTE (MODELO ML)



st.caption("""
La lectura pedagógica se basa en un umbral cuidadosamente definido.
El modelo solo señala desajuste cuando la probabilidad es alta,
para evitar alertas innecesarias y respetar los ritmos naturales del desarrollo.
""")




st.write(round(riesgo_base, 2))

st.divider()

# =========================================================
# ESCENARIO DIGITAL — IED
# =========================================================
st.subheader("Entorno digital (IED)")

st.markdown(""" <div class="montessori-box">
<p>
Las pantallas forman parte del mundo actual, pero su uso debe estar equilibrado.
</p>
<p>
Aquí puedes simular **cómo distintos entornos digitales** pueden influir
en el desarrollo, sin cambiar al niño.
</p>
</div>""", unsafe_allow_html=True)

ied_options = {
    0.0: "Uso digital muy desequilibrado, con impacto negativo.",
    0.25: "Uso frecuente con poco acompañamiento.",
    0.5: "Uso moderado, con equilibrio variable.",
    0.75: "Uso mayormente equilibrado y acompañado.",
    1.0: "Uso digital consciente, educativo y bien integrado."
}

ied_label = st.selectbox(
    "¿Cómo describirías el entorno digital?",
    options=list(ied_options.keys()),
    format_func=lambda x: ied_options[x],
    help="Considera la frecuencia, el tipo de uso y el acompañamiento adulto."
)

ied = ied_label

riesgo_ajustado = ajustar_riesgo_por_ied(riesgo_base, ied)

st.subheader("Riesgo considerando el entorno digital")
st.write(round(riesgo_ajustado, 2))

st.divider()

# =========================================================
# MENSAJE FINAL
# =========================================================

st.markdown(
    """
    ### Reflexión final

    En la pedagogía Montessori, el desarrollo no se mide por resultados
    inmediatos, sino por la **coherencia entre el niño y su entorno**.

    Esta herramienta no ofrece diagnósticos ni etiquetas.
    Ofrece **información para observar, comprender y acompañar mejor**.

    Cuando surge una señal de desajuste, la pregunta no es:
    *¿qué le pasa al niño?*,  
    sino:
    *¿qué necesita el ambiente para acompañarlo mejor?*
    """
)

