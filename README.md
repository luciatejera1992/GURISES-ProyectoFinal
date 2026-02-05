# GURISES - Proyecto Final

## Objetivo general

Gurises es un proyecto de análisis de datos aplicado a la educación infantil y adolescente, cuyo objetivo principal es comprender y acompañar el desarrollo de niños y adolescentes desde una mirada integral, respetuosa y basada en la pedagogía Montessori.

El proyecto nace de una convicción clara:
los datos y la tecnología pueden ser herramientas valiosas para la educación solo si se utilizan con responsabilidad, sensibilidad pedagógica y respeto por los procesos naturales del desarrollo humano.

En lugar de centrarse exclusivamente en el rendimiento académico o en resultados numéricos, Gurises pone el foco en la relación entre el niño, su entorno, su autonomía y su bienestar, entendiendo que el aprendizaje genuino surge cuando estas dimensiones se encuentran en equilibrio.

## Qué hace el proyecto

El proyecto integra distintas etapas y herramientas:

- Análisis de datasets educativos reales, relacionados con hábitos de estudio, entorno, bienestar y uso de tecnología.
- Construcción de índices pedagógicos que sintetizan dimensiones clave del desarrollo:
  - **ISEE** — Entorno educativo
  - **IAA** — Autonomía y autorregulación
  - **IBE** — Bienestar
  - **IED** — Equilibrio digital
- Análisis exploratorio y visualización de datos en Power BI.
- Modelado con machine learning para identificar señales de desajuste entre dimensiones.
- Aplicación interactiva en Streamlit orientada a familias, educadores e instituciones.

## Estructura del repositorio

```
GURISES-ProyectoFinal/
├── app.py                 # Aplicación Streamlit principal
├── requirements.txt       # Dependencias del proyecto
├── LICENSE                # Licencia MIT
├── data/
│   ├── StudentPerformanceFactors.csv   # Dataset principal (factores de rendimiento)
│   ├── powerbi_dataset.xlsx            # Dataset exportado para Power BI
│   ├── screen_time.csv                 # Datos de tiempo de pantalla
│   └── ied_by_age_group.csv            # Índice de equilibrio digital por grupo etario
├── models/
│   ├── modelo_ml.pkl      # Modelo de clasificación (Logistic Regression)
│   ├── kmeans.pkl         # Modelo de clustering (KMeans k=4)
│   └── scaler.pkl         # Escalador para clustering (StandardScaler)
├── notebooks/
│   ├── Factors1.ipynb     # EDA + ingeniería de features + ML + clustering
│   └── EDAscreentime.ipynb # EDA de tiempo de pantalla + cálculo de IED
└── assets/
    └── logo.png           # Logo de la marca
```

## Instalación

**Requisitos:** Python 3.10+

```bash
git clone https://github.com/GURISES/GURISES-ProyectoFinal.git
cd GURISES-ProyectoFinal
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

### Aplicación Streamlit

```bash
streamlit run app.py
```

### Notebooks

Explora los notebooks para el análisis detallado:

1. `notebooks/EDAscreentime.ipynb` — Análisis de tiempo de pantalla y cálculo de IED
2. `notebooks/Factors1.ipynb` — EDA completo, construcción de índices y modelado ML

> **Nota:** Ejecuta `EDAscreentime.ipynb` antes de `Factors1.ipynb`, ya que este último importa el archivo `ied_by_age_group.csv` generado por el primero.

## Metodología

El proyecto utiliza técnicas de análisis de datos y aprendizaje automático para:

- Identificar factores clave que afectan el desarrollo infantil y adolescente.
- Crear índices sintéticos (ISEE, IAA, IBE, IED) que cuantifiquen estos factores.
- Detectar señales de desajuste pedagógico mediante un modelo de clasificación.
- Agrupar perfiles educativos mediante clustering (KMeans).

## Visualizaciones

Incluye dashboards interactivos creados con Power BI:

[Abrir Panel de Power BI](https://app.powerbi.com/groups/me/reports/d113795a-6f16-4f96-9f8c-31c344a3c925/3b85740245d2d995ceba?experience=power-bi)

## Contacto

Para preguntas o sugerencias:
- **Nombre**: Lucía Tejera
- **Email**: luciatejera1992@gmail.com

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).
