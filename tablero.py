import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.encoding import CountFrequencyEncoder
from PIL import Image
st.set_page_config(
    page_title="Tablero de Análisis y Modelado de Datos",
    layout="wide",
)

# Título del Tablero
st.title("Tablero de Análisis y Modelado de Datos")

st.markdown(
    """
**Prueba técnica:**  
- El siguiente reporte presenta los resultados del análisis y modelamiento de datos
asociados a la conversión alimenticia y como esta puede ser interpretada bajo un conjunto
de predictores. *Detalles de los calculos y mayor descripción de las variables se podrá encontrar
en el notebook de exploración y modelado*.

Iniciamos cargando los datos raw:
"""
)


# Cargar datos
@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_datos():
    df = pd.read_csv(
        "Prueba de modelamiento para Cientifico de Datos 2023.csv",
        sep=";",
        encoding="latin-1",
    )
    return df


df = cargar_datos()

# Identificar columnas categóricas y numéricas
categorical_cols = ["fabricaAlimento", "granja_anonym", "genetica", "asesor"]
numerical_cols = ["pesoInicial", "pesoFinal", "semesreSalida", "mortalidad"]

# Barra lateral para filtros de visualización
st.sidebar.header("Filtros de Visualización")

# Barra lateral para filtros
st.sidebar.header("Filtros")

# Filtro por especie
especies = st.sidebar.multiselect(
    "Selecciona tipo de genetica:",
    options=df["genetica"].unique(),
    default=df["genetica"].unique(),
)

anioSalida = st.sidebar.multiselect(
    "Selecciona por año de salida:",
    options=df["anioSalida"].unique(),
    default=df["anioSalida"].unique(),
)

# Filtro por rango de sepal length
min_initial_weight = float(df["pesoInicial"].min())
max_initial_weight = float(df["pesoInicial"].max())

sepal_length = st.sidebar.slider(
    "Rango de pesoInicial:",
    min_value=min_initial_weight,
    max_value=max_initial_weight,
    value=(min_initial_weight, max_initial_weight),
)

costo_por_kg = st.sidebar.selectbox(
    "Valor kilogramo alimento: ",
    options=[2500, 2000, 5000, 0],
    index=0,  # Índice del valor por defecto (0 corresponde a 0)
)

peso_inicial = st.sidebar.selectbox(
    "Peso inicial: ",
    options=[20, 30, 40, 50],
    index=0,  # Índice del valor por defecto (0 corresponde a 0)
)
peso_final = st.sidebar.selectbox(
    "Peso final: ",
    options=[220, 210, 230, 30],
    index=0,  # Índice del valor por defecto (0 corresponde a 0)
)
poblacion = st.sidebar.selectbox(
    "Población: ",
    options=[200000, 210000, 18000, 10],
    index=0,  # Índice del valor por defecto (0 corresponde a 0)
)


# Aplicar filtros al DataFrame
filtro = (
    (df["genetica"].isin(especies))
    & (df["anioSalida"].isin(anioSalida))
    & (df["pesoInicial"] >= sepal_length[0])
    & (df["pesoInicial"] <= sepal_length[1])
)

df_filtrado = df[filtro]

# Mostrar tabla de datos
st.header("Datos de trabajo")
st.dataframe(df)

# Diseño del Tablero
# Utilizar columnas para organizar el contenido
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Conversión Alimenticia")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtrado["conversionAlimenticia"],
        bins=20,
        kde=True,
        ax=ax1,
        color="skyblue",
        edgecolor="black",
    )

    # Calcular la media
    mean_conversion = df_filtrado["conversionAlimenticia"].mean()

    # Agregar línea vertical en la media
    ax1.axvline(mean_conversion, color="red", linestyle="--", linewidth=2)

    # Agregar etiqueta para la media
    ax1.text(
        mean_conversion + 0.1,
        ax1.get_ylim()[1] * 0.9,
        f"Mean: {mean_conversion:.2f} kilos",
        color="red",
        ha="center",
        fontsize=12,
        backgroundcolor="white",
    )

    # Configurar etiquetas y título
    ax1.set_xlabel("Conversión Alimenticia")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución de la Conversión Alimenticia con la Media")

    st.pyplot(fig1)
    st.markdown(
        """
            A nivel global (sin usar los filtros globales, la conversión alimenticia muestra una
            distribucción sesgada o inclinada hacia menores valores de conversión, con un valor
            medio en la muestra trabajada de  2.24 Kilos.
            """
    )

with col2:
    # st.header("Relación Sepal vs. Petal")
    # fig2, ax2 = plt.subplots()
    # sns.scatterplot(data=df_filtrado, x='edadFinal',  y='conversionAlimenticia', ax=ax2)
    # st.pyplot(fig2)
    # Crear columnas internas para centrar la tabla

    st.subheader("Estadísticas de Conversión Alimenticia")

    st.dataframe(df_filtrado.describe())
    st.markdown(
        """
            Dentro de las variables numéricas podríamos inferir que la edad de los animales
            se mide en semanas, donde el proceso de medición inicia en promedio cuando los
            animales poseen 72 semanas  y concluye su vida en promedio a las 157 semanas. Estos
            animales en promedio empiezan con un peso de 31 kilos y concluyen con un peso promedio
             de 113 kilos
            La mortalidad tiene un promedio de 1.88% con un máximo de 10%.

            """
    )

# Añadir Box Plots
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>Box Plots: variables categóricas</h2>",
    unsafe_allow_html=True,
)
# Crear columnas para los box plots
box_col1, box_col2 = st.columns(2)

with box_col1:
    st.subheader("Conversión por fabrica de alimento")
    fig3, ax3 = plt.subplots()
    sns.violinplot(
        x="fabricaAlimento",
        y="conversionAlimenticia",
        data=df_filtrado,
        ax=ax3,
        palette="Set3",
        inner="quartile",
    )
    ax3.set_xlabel("Fabrica de Alimento")
    ax3.set_ylabel("Conversión")
    st.pyplot(fig3)
#
with box_col2:
    st.subheader("Conversión por tipo de genetica")
    fig4, ax4 = plt.subplots()
    sns.violinplot(
        x="genetica",
        y="conversionAlimenticia",
        data=df_filtrado,
        ax=ax4,
        palette="Set3",
        inner="quartile",
    )
    ax4.set_xlabel("genetica")
    ax4.set_ylabel("Conversión")
    st.pyplot(fig4)

with box_col1:

    asesor = [
        "rt_18",
        "rt_19",
        "rt_6",
        "rt_0",
        "rt_17",
        "rt_12",
        "rt_2",
        "rt_16",
        "rt_0",
        "rt_20",
    ]
    df_asseror = df[df["asesor"].isin(asesor)]

    st.subheader("Conversión por los principales asesores")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="asesor", y="conversionAlimenticia", data=df_asseror, ax=ax3)
    ax3.set_xlabel("Asesor")
    ax3.set_ylabel("Conversión")
    st.pyplot(fig3)
#
with box_col2:
    granjas = [
        "gr_48",
        "gr_2",
        "gr_18",
        "gr_70",
        "gr_0",
        "gr_9",
        "gr_2" "gr_36",
        "gr_54",
        "gr_44",
        "gr_47",
        "gr_35",
    ]
    df_granja = df[df["granja_anonym"].isin(granjas)]

    st.subheader("Conversión por Granja")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="granja_anonym", y="conversionAlimenticia", data=df_granja, ax=ax4)
    ax4.set_xlabel("Granja")
    ax4.set_ylabel("Conversión")
    st.pyplot(fig4)

st.markdown(
    """
            - Las fabricas de alimento que mejor performance en conversión han tenido, es decir, el valor
            medio en conversión de la fabrica, comparado con el valor medio de toda la muestra de datos son pp_0
            y pp_8.  Caso opuesto se puede ver en la genetica de los animales, donde las categorías lg_6 y lg_2 
            poseen un valor medio superior al de la muestra analizada.
            - En Adición, aunque en un mejor grado, los asesores rt_0 y rt_2 presentan un performance superior
            que los demás. En el caso de las granjas, se puede evidenciar un comportamiento algo atípico en el
            grupo de granjas para la gr_18 donde su valor medio degrada la conversión. **De igual forma se observan 
            valores atípicos en la granja 48, pero estos valores mejoran el performance de conversión y la granja 2 
            que posee un valor medio de conversión por debajo del valor medio de la población.**
            """
)
# Dispersión

# Añadir Box Plots
st.markdown(
    "<h2 style='text-align: center; color: #0D47A1;'>Relación de dispersion: variables numéricas</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """En los siguientes gráficos cada punto representa 
            un lote de cerdo y como la conversión alimenticia cambia en función de
            algún predictor numérico. Los puntos en rojo, son valores atípicos y están calculados
            como aquellos datos que superan más de tres desviaciones estándar de la media.
            """
)

# TODO: scater
scatter_col1, scatter_col2 = st.columns(2)
df_filtrado["z_score_duracionCeba"] = stats.zscore(df_filtrado["duracionCeba"])

# Identificar outliers
outliers = np.abs(df_filtrado["z_score_duracionCeba"]) > 3
with scatter_col1:
    st.subheader("Dispersión de Duración de Ceba y Conversión Alimenticia")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    ax_scatter.scatter(
        df_filtrado.loc[~outliers, "duracionCeba"],
        df_filtrado.loc[~outliers, "conversionAlimenticia"],
        alpha=0.6,
        label="Datos normales",
        color="blue",
    )

    ax_scatter.scatter(
        df_filtrado.loc[outliers, "duracionCeba"],
        df_filtrado.loc[outliers, "conversionAlimenticia"],
        color="red",
        label="Valores atípicos",
        alpha=0.8,
    )

    # Configuración del gráfico
    ax_scatter.set_title(
        "Dispersión de Duración de Ceba y Conversión Alimenticia (con Valores Atípicos)"
    )
    ax_scatter.set_xlabel("Duración de Ceba (kilos)")
    ax_scatter.set_ylabel("Conversión Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """Lotes de cerdos con menos de 50 semanas de Ceba son casos atípicos, adiciones la conversión
    es superior al valor medio de la muestra total. A nivel descriptivo se observa
    que una conversión por debajo de la media se obtiene con al menos 105 semanas de ceba, 
    por encima de estas, la conversión se degrada
        """
    )

with scatter_col2:
    df_filtrado["z_score_edad_final"] = stats.zscore(df_filtrado["edadFinal"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_score_edad_final"]) > 3
    st.subheader("Dispersión de edad final y Conversión Alimenticia")

    # Crear el scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    # Scatter plot de datos normales
    ax_scatter.scatter(
        df_filtrado.loc[~outliers, "edadFinal"],
        df_filtrado.loc[~outliers, "conversionAlimenticia"],
        alpha=0.6,
        label="Datos normales",
        color="blue",
    )

    # Scatter plot de outliers
    ax_scatter.scatter(
        df_filtrado.loc[outliers, "edadFinal"],
        df_filtrado.loc[outliers, "conversionAlimenticia"],
        color="red",
        label="Valores atípicos",
        alpha=0.8,
    )

    # Configuración del gráfico
    ax_scatter.set_title(
        "Dispersión de edadFinal y Conversión Alimenticia (con Valores Atípicos)"
    )
    ax_scatter.set_xlabel("EdadFinal (meses)")
    ax_scatter.set_ylabel("Conversión Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
                La edad esta lineal mente relacionada con la conversión, por encima de 170 semanas
                de edad, los lotes de cerdos degradan el factor de conversión. Se evidencia también
                los valores atípicos en cerdos con edad final menor a 120 semanas los cuales para ser tan 
                jóvenes degradaron su conversión seguro por algún factor externo.
            """
    )


scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    df_filtrado["z_score_mortalidad"] = stats.zscore(df_filtrado["mortalidad"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_score_mortalidad"]) > 3
    st.subheader("Dispersión mortalidad y Conversión Alimenticia")

    # Crear el scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    # Scatter plot de datos normales
    ax_scatter.scatter(
        df_filtrado.loc[~outliers, "mortalidad"],
        df_filtrado.loc[~outliers, "conversionAlimenticia"],
        alpha=0.6,
        label="Datos normales",
        color="blue",
    )

    # Scatter plot de outliers
    ax_scatter.scatter(
        df_filtrado.loc[outliers, "mortalidad"],
        df_filtrado.loc[outliers, "conversionAlimenticia"],
        color="red",
        label="Valores atípicos",
        alpha=0.8,
    )

    # Configuración del gráfico
    ax_scatter.set_title(
        "Dispersión de mortalidad y Conversión Alimenticia (con Valores Atípicos)"
    )
    ax_scatter.set_xlabel("mortalidad")
    ax_scatter.set_ylabel("Conversión Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
            En general existe baja mortalidad, pero aquellos lotes donde la mortalidad supera un 7%, 
            se tienen como anomalías.
        """
    )

with scatter_col2:
    df_filtrado["z_pesoInicial"] = stats.zscore(df_filtrado["pesoInicial"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_pesoInicial"]) > 3
    st.subheader("Dispersión pesoFinal y Conversión Alimenticia")

    # Crear el scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    # Scatter plot de datos normales
    ax_scatter.scatter(
        df_filtrado.loc[~outliers, "pesoInicial"],
        df_filtrado.loc[~outliers, "conversionAlimenticia"],
        alpha=0.6,
        label="Datos normales",
        color="blue",
    )

    # Scatter plot de outliers
    ax_scatter.scatter(
        df_filtrado.loc[outliers, "pesoInicial"],
        df_filtrado.loc[outliers, "conversionAlimenticia"],
        color="red",
        label="Valores atípicos",
        alpha=0.8,
    )

    # Configuración del gráfico
    ax_scatter.set_title(
        "Dispersión pesoInicial y Conversión Alimenticia (con Valores Atípicos)"
    )
    ax_scatter.set_xlabel("pesoInicial (kilos)")
    ax_scatter.set_ylabel("Conversión Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
                Lotes con pesos iniciales superior a 45 kilos se consideran anomalías, adicional degradan
                la conversión.
            """
    )


st.markdown(
    """
                **Conclusiones generales a nivel descriptivo**.
                 - La conversión alimenticia tiene un promedio de 2.24 con un rango entre 1.80 y 2.79.
                 - El peso inicial promedio es 31.28 kg, y el peso final promedio es 113.90 kg.
                 - Los lotes de cerdo deberían de iniciar con un peso entre 25 y 40 Kg, para obtener al
                   final del proceso, conversiones por debajo del valor medio de la muestra analizada.
                 - El asesor rt_0 y rt_2 han mejorado el factor de conversión con respecto a los demás. 
                   Seria interrelate revisar el volumen de lotes y cerdos en la cual se ha obtenido mejoras.
                 - Mortalidad tiene un promedio de 1.88% con un máximo de 10%.

    """
)
numerical_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_columns].corr()
st.header("Matriz de Correlación")

st.markdown(
    """
Esta matriz de correlación muestra la relación entre las variables numéricas del dataset. 
"""
)

# Crear y mostrar el heatmap
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    square=True,
    ax=ax,
)
ax.set_title("Matriz de Correlación entre Variables Numéricas", fontsize=16, pad=20)
plt.tight_layout()

st.pyplot(fig)

st.markdown(
    """**Correlaciones**:
El mapa de calor muestra correlaciones entre variables Peso inicial y peso final tienen una correlación
 positiva fuerte, lo cual es esperado.
Adicional y como es de esperar, el peso y edad inicial/final, 
respectivamente presentan correlation positiva fuerte, 
es por tal motivo que en un modelo de machine learning
no es bueno tenerlas todas, para evitar colinealidades.
  Conversión alimenticia presenta relaciones moderadas con variables como mortalidad y peso inicial.
    """
)

# Título de la aplicación

# Subir una imagen
image_path = "img.png"  # Cambia esto por la ruta real de tu imagen

image = Image.open(image_path)

    # Mostrar la imagen
st.image(image, caption="Imagen cargada desde la carpeta", use_container_width=True)

st.markdown(
    """Un resumen descriptivo de la conversión alimenticia se puede dar con la anterior 
    grafica, la cual fue obtenidad en el `reporte_sweetviz.html` en ``exporation.ipynb``:
    
    - Tenemos nuestra variable de interés que es la conversión con un valor medio de 2.23, es decir,
      por cada 2.23 kg que come el cerdo, este incrementa en 1 su peso.
    - La Razón de Correlación mide la fuerza de asociación entre una variable 
      categórica y una variable continua, nos muestra que la granja y la empresa y el asesor,
      fabrica de alimento están altamente asociadas, mientras que el semestre de salida practicamente
      no tiene efecto en el comportamiento de la conversión. Esto se recalcará más adelante en la sección
      de modelado.
    """
)

# Sección de Modelado
st.header("🔧 Entrenamiento y Evaluación del Modelo")

st.markdown(
    """
**Descripción del Modelo de Conversión Alimenticia:**  
En esta sección, entrenamos un modelo de regresión para predecir la conversión 
alimenticia basada en diversas características como la fábrica de alimento, granja,
genética, peso inicial y final, asesoría y mortalidad ¿Por qué? No es solo por que nos lo indica la prueba,
es por que la razón de asociación y factor de correlación nos lo indica.
 
La conversión alimenticia es un indicador clave para evaluar la eficiencia alimentaria 
de los animales en la producción ganadera y el objetivo final es entender como y que valores
o conjuntos de valores afectan esta conversión.

Para esta tarea primero debemos transformar la data raw, aunque ya esta limpia, 
debemos codificar las variables categóricas y normalizar las variables numéricas para obtener
un mejor performance en los modelos.
"""
)

# Función para entrenar el modelo


st.header(" Estructura del Pipeline de Preprocesamiento")
encoder_freq = CountFrequencyEncoder(
    encoding_method="frequency",
    variables=categorical_cols,
    ignore_format=True,
)
scaler = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("categoricals", encoder_freq, categorical_cols),
        ("numericals", scaler, numerical_cols),
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Mostrar los pasos del pipeline
for name, step in pipeline.named_steps.items():
    st.subheader(f"Paso: {name}")
    st.write(step)

st.markdown(
    """
Con la transformaciones mencionadas, nuestra data raw se convierte en:
"""
)

# Mostrar tabla de datos
st.header("Datos de procesados")


@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_procesados():
    df = pd.read_csv("datos_transformados.csv")
    return df


st.dataframe(cargar_procesados())


st.markdown(
    """
A continuación se entrena diferentes modelos de regresión usando un AutoML, en este caso se uso pycaret y los
detalles se pueden observar en el notebook de trabajo, pero aca mostraremos las modelos entrenados
y los valores de las métricas obtenidas en el conjunto de test:
"""
)


@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_datos_modelos_entrenados():
    df = pd.read_csv("comparacion_modelos.csv")
    return df


# Mostrar tabla de datos
st.header("Comparación de modelos entrenados")
st.dataframe(cargar_datos_modelos_entrenados())

st.markdown(
    """ **Cuál modelo escoger?**

En base al valor de las métricas y la posterior interpretación e importancia de los predictores
se escoje el Random Forest ya que reduce el valor de la MAE.
 
 MAE no penaliza excesivamente los errores grandes, lo que significa 
 que refleja de manera más equilibrada el rendimiento del modelo
  sin ser dominado por unos pocos outliers.
  
Los valores SHAP proporcionan una explicación local (por instancia) y global (promedio), por lo que
 tener una métrica como MAE facilita correlacionar la magnitud de 
 los errores con las contribuciones de las características.
 
Al entender que el MAE refleja el error promedio en
 la misma escala que los datos,
 puedes relacionar más directamente cómo las 
 características influyen en este error promedio.
"""
)
st.markdown("Detalles del paso a paso, revisión de residual y error, se pueden encontrar"
            "en `pycaret.ipynb`")

st.header("Interpretación de resultados en base a los Shap values")

@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_shap_values():
    return joblib.load("shap_values.joblib")


shap_values = cargar_shap_values()




st.subheader("Gráfico mean SHAP ")
fig8, ax8 = plt.subplots()
shap.plots.bar(shap_values)
st.pyplot(fig8)

st.markdown(
    """Para cada característica en el modelo entrenado, se otorga el 
    valor de SHAP medio absoluto. La granja y el asesor son altamente
     importantes para el proceso de predicción, tal como vimos
    en la parte inicial con el análisis de box  y con el radio de correlación."""
)

st.subheader("Gráfico Beeswarm de SHAP")
fig7, ax7 = plt.subplots()
shap.plots.violin(shap_values, plot_type="layered_violin")
#shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig7)

st.markdown(
    """Para saber la importancia de las predictoras, usaremos los 
diagramas de SHAP ya que permite ver los aportes de cada feature en el modelo
entrenado. Para este modelo la feature granja es la que contribuye más en la 
predicción del modelo, donde valores altos en este parámetro dan un valor negativo
en el SHAP indicando una reducción en la variable predictora o conversión, lo que 
nos favorece para el negocio.
En esta misma feature, la concentración de valores se tiene en un valor positivo 
de SHAP, indicando que en general valores menores en granja aumenta la conversión.
¿Que significa un valor alto o pequeño en la granja? Depende de la transformación 
de nuestros datos, un valor alto es que se tiene alta frecuencia, según nuestra 
transformación, por tanto la granja más frecuente fue gr_48 y la gr_2, categorías 
que en el box plot centramos su atención.

La mortalidad es otra feature fácil de entender, valores pequeños en mortalidad 
reducen el SHAP y por tanto la conversión, como es de esperar y lo vimos en el 
análisis descriptivo. Existe un valor alto en mortalidad que incrementa la conversión,
estos son los punto extremos que encontramos en el análisis descriptivo.

Para los asesores, vemos una densidad de puntos de alto valor en la zona positiva
del SHAP, esto indica que los asesores con mayor frecuencia están levemente incrementando
la conversión.
En el caso de la genética, vemos que valores grandes en la genética, la categoría
que más se repite tiene a tener un valor negativo en SHAP, y por tanto reducir 
la conversión. Mientras que en general, valores pequeños en la genética, aumentan
la conversión, afortunadamente mayoritariamente se tiene solo una genética.

Si el peso inicial es pequeño, esto mejora al final la conversión. Algo opuesto
al peso final.

En el caso de el semestre de salida, no hay efecto si se saca en el primer
o segundo semestre la producción. Ya que la concentración de puntos están en 0.


Ahora, para ver que tanto contribuye una variable predictora en la predicción,
tendremos los siguientes gráficos tanto a nivel global como a nivel particular.
De destacar que los diferenciales en que contribuyen cada predictor son menores que 
el error absoluto (MAE).

"""
)


st.subheader("Mapa de calor de SHAP")
fig7, ax7 = plt.subplots()
shap.plots.heatmap(shap_values)
#shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig7)



st.subheader("Gráfico Waterfall de la Predicción Seleccionada")
index = st.number_input(
    "Ingrese el número de la predicción (0 a 2524):",
    min_value=0,
    max_value=len(shap_values) - 1,
    step=1,
    value=0,
)


st.markdown(
    """¿Qué contribución otorga cada predictor en el resultado final de la
    predicción y por que esta se aleja del valor promedio de la muestra global?
    La siguiente grafica nos permite estimar la  contribución diferencial en cada predictor, 
    que aleja o acerca el valor predicho del valor global.
    En interpretación con el negocio tener una conversión menor (por debajo de la media)
    es favorable para el negocio, por tanto en el valor por defecto, el peso inicial y la
    fabrica de alimento, han contribuido a reducir la conversión, mientras el asesor y la
    mortalidad han ayudado a aumentarla."""
)
# Gráfico Waterfall basado en la entrada del usuario
fig9, ax9 = plt.subplots()
shap.plots.waterfall(shap_values[int(index)], show=False)
st.pyplot(fig9)



# Gráfico Force basado en la entrada del usuario
st.subheader("Gráfico Force de la Predicción Seleccionada")
fig10 = shap.plots.force(shap_values[int(index)], matplotlib=True)
st.pyplot(fig10)
# st.subheader("Gráfico Waterfall de la Primera Predicción")
# fig6, ax6 = plt.subplots()
# shap.plots.waterfall(shap_values[int(index)], show=False)
# st.pyplot(fig6)
st.markdown(
    """Y en este caso vemos la fuerza y el valor en la variable predictora
    que ocasionó el cambio. Debemos de traducir los valores de la features por medio de
    la transformación realizada. Ejemplo, el asesor=0.038 representa una frecuencia,
    del asesor rt_2."""
)



# Extraer los valores SHAP asociados a las columnas categóricas
categorical_columns = [
    col for col in cargar_procesados().columns if col in categorical_cols
]
# Extraer los valores SHAP asociados a las columnas categóricas
categorical_shap_values = pd.DataFrame(
    shap_values.values[
        :, [cargar_procesados().columns.get_loc(col) for col in categorical_columns]
    ],
    columns=categorical_columns,
)

# Calcular los valores mínimos y máximos por variable
categorical_min = categorical_shap_values.min(axis=0)
categorical_max = categorical_shap_values.max(axis=0)

# Crear un DataFrame resumen para mostrar los resultados
categorical_summary = pd.DataFrame(
    {
        "Categoría": categorical_min.index,
        "SHAP Min": categorical_min.values,
        "SHAP Max": categorical_max.values,
    }
)


conversion_mean = df["conversionAlimenticia"].mean()

peso_ganado = peso_final - peso_inicial
categorical_summary["conv_max"] = conversion_mean + categorical_summary["SHAP Max"]
categorical_summary["conv_min"] = conversion_mean + categorical_summary["SHAP Min"]
categorical_summary["consumo_max"] = categorical_summary["conv_max"] * peso_ganado
categorical_summary["consumo_min"] = categorical_summary["conv_min"] * peso_ganado
categorical_summary["ahorro_consumo"] = (
    categorical_summary["consumo_max"] - categorical_summary["consumo_min"]
)
ahorro_dinero = categorical_summary["ahorro_consumo"] * costo_por_kg
categorical_summary["ahorro_dinero"] = ahorro_dinero

st.header("Aplicación")
st.markdown(
    "5.	Suponga que el kilo de alimento cuesta $2.500 pesos."
    " En las siguientes variables categóricas: asesor,"
    " fabrica Alimento, granja y genetica, "
    "identifique el peor y el mejor nivel o categoría"
    " según su efecto marginal sobre la conversión."
    " Estime cuánta diferencia de alimento se "
    "ahorra el mejor nivel respecto del peor sobre "
    "una población de 200.000 cerdos  que pasa de 30"
    " a 220 kg de peso vivo y valorice dicha diferencia."
)

st.markdown(
    "La respuesta se construirá en base a los valores SHAP, el menor  y mayor "
    "valor SHAP en cada predictor mencionado, representa el mejor y peor valor e contribución"
    "que cada predictor respectivamente puede otorgar a la predicción de conversión."
    "Recordemos que una conversión menor es"
    "mejor para el negocio, por tanto un valor SHAP menor indicaría una menor contribución"
    "en alcanzar el valor medio de la conversión.")

    
st.markdown(
    "Tomando entonces:conv_max/min = conversión_mean -/+ SHAP Min/Max"
    "y como el consumo o alimento otorgado al animal es: conv * peso ganado, "
    "tenemos el peso por animal que se ha consumido, y por tanto el peso total "
    "en todo el lote y su valor."
)
st.dataframe(categorical_summary)

st.markdown(
    "Por último, vemos que la granja es la que aporta más en el ahorro"
    "en alimento y por tanto en dinero, es decir, eligiendo las granjas que"
    "poseen un valor medio en conversión inferior a la media de la muestra "
    "analizada, el modelo indica que la conversión disminuye aumentando la"
    "rentabilidad en ahorro de gasto alimenticio."
)

st.header("¿Qué más se podria intentar responder o modelar con los datos?")
st.markdown(
    """Algunas ideas serian:
    
    - ¿Existen diferencias significativas en la conversión alimenticia entre
       lotes según categorías como fábrica de alimento, genética, o granja?
    
    - ¿Qué combinación de las variables categóricas lleva a la conversión alimenticia más baja?
    
    - ¿Es posible identificar prácticas en granjas o líneas genéticas que reduzcan 
       la mortalidad y, por ende, mejoren la conversión alimenticia?
    - ¿Hay un peso final óptimo que minimice la conversión alimenticia 
       antes de que se estabilice o comience a aumentar?""")

# valor_kilo_alimento
# Footer
st.markdown("---")
st.markdown("**Hecho  por [Juan David Rincón](jdrincone@gmail.com)**")
