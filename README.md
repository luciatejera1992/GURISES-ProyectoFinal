# GURISES - Proyecto Final

## Objetivo general
Gurises es un proyecto de análisis de datos aplicado a la educación infantil y adolescente, cuyo objetivo principal es comprender y acompañar el desarrollo de niñ@s y adolescentes desde una mirada integral, respetuosa y basada en la pedagogía Montessori.

El proyecto nace de una convicción clara:
los datos y la tecnología pueden ser herramientas valiosas para la educación solo si se utilizan con responsabilidad, sensibilidad pedagógica y respeto por los procesos naturales del desarrollo humano.

En lugar de centrarse exclusivamente en el rendimiento académico o en resultados numéricos, Gurises pone el foco en la relación entre el niño, su entorno, su autonomía y su bienestar, entendiendo que el aprendizaje genuino surge cuando estas dimensiones se encuentran en equilibrio.


## Qué hace el proyecto 

El proyecto integra distintas etapas y herramientas:

Análisis de datasets educativos reales, relacionados con hábitos de estudio, entorno, bienestar y uso de tecnología.

Construcción de índices pedagógicos que sintetizan dimensiones clave del desarrollo:

Entorno educativo

Autonomía y autorregulación

Bienestar

Equilibrio digital

Análisis exploratorio y visualización de datos en Power BI, para mostrar de forma transparente y comprensible los patrones presentes en los datos.

Modelado con machine learning, utilizado de forma ética para identificar posibles señales de desajuste entre dimensiones, siempre contextualizadas pedagógicamente.

Desarrollo de una aplicación en Streamlit, orientada a familias, educadores e instituciones, que traduce los resultados técnicos en mensajes claros, accesibles y alineados con la pedagogía Montessori.

## Contenido del repositorio

- **`app.py`**: Archivo principal para la ejecución del proyecto.
- **`powerbi_dataset.csv` y `powerbi_dataset.xlsx`**: Conjuntos de datos utilizados para análisis y visualización.
- **`EDAscreemtime.ipynb` y `Factors1.ipynb`**: Notebooks de Jupyter para análisis exploratorio de datos (EDA).
- **`modelo_ml.pkl`**: Modelo de aprendizaje automático entrenado.
- **`kmeans.pkl`**: Modelo de clustering K-Means.
- **`scaler.pkl`**: Escalador utilizado para preprocesamiento de datos.
- **`features.pkl`**: Características seleccionadas para el modelo.
- **`screen_time.csv` y `screen_time_ied.csv`**: Datos relacionados con el tiempo de pantalla y el índice de desarrollo educativo.
- **`StudentPerformanceFactors.csv`**: Datos sobre factores que afectan el rendimiento estudiantil.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/GURISES-ProyectoFinal.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd GURISES-ProyectoFinal
   ```
3. Instala las dependencias necesarias (asegúrate de tener `pip` instalado):
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

1. Ejecuta el archivo principal para iniciar el análisis:
   ```bash
   python app.py
   ```
2. Explora los notebooks para un análisis detallado y visualización de datos:
   - `EDAscreemtime.ipynb`
   - `Factors1.ipynb`

## Aplicación Interactiva

El proyecto incluye una aplicación interactiva desarrollada con **Streamlit** que permite analizar el desajuste educativo de manera dinámica. Esta herramienta facilita la exploración de los datos y la visualización de los índices sintéticos creados, proporcionando una experiencia intuitiva para los usuarios interesados en comprender los factores que afectan el desarrollo infantil y adolescente.

Para ejecutar la aplicación interactiva, utiliza el siguiente comando:
```bash
streamlit run app.py
```

## Metodología

El proyecto utiliza técnicas de análisis de datos y aprendizaje automático para:
- Identificar factores clave que afectan el desarrollo infantil y adolescente.
- Crear índices sintéticos que cuantifiquen estos factores.
- Evaluar el impacto del desajuste educativo en el desarrollo integral.

## Visualizaciones

Incluye dashboards interactivos creados con Power BI para explorar los datos y resultados de manera intuitiva.

## Acceso al Panel de Power BI

Puedes acceder al dashboard interactivo de Power BI haciendo clic en el siguiente botón:

[Abrir Panel de Power BI]

https://app.powerbi.com/groups/me/reports/d113795a-6f16-4f96-9f8c-31c344a3c925/3b85740245d2d995ceba?experience=power-bi

## Contacto

Para preguntas o sugerencias:
- **Nombre**: Lucía Tejera
- **Email**: luciatejera1992@gmail.com

---

😊


