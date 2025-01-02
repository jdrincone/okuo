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

st.set_page_config(
    page_title="Tablero de An�lisis y Modelado de Datos",
    layout="wide",
)

# T�tulo del Tablero
st.title("=� Tablero de An�lisis y Modelado de Datos")

st.markdown(
    """
**Prueba tecnica:**  
- El siguiente reporte presenta los resultados del an�lisis y modelamiento de datos
asociados a la conversi�n alimenticia y como esta puede ser interpretada bajo un conjunto
de predictores. Detalles de los calculos y mayor descripci�n de las variables se podr� encontrar
en el notebook de exploraci�n y modelado.

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

# Identificar columnas categ�ricas y num�ricas
categorical_cols = ["fabricaAlimento", "granja_anonym", "gen�tica", "asesor"]
numerical_cols = ["pesoInicial", "pesoFinal", "semesreSalida", "mortalidad"]

# Barra lateral para filtros de visualizaci�n
st.sidebar.header("Filtros de Visualizaci�n")

# Barra lateral para filtros
st.sidebar.header("Filtros")

# Filtro por especie
especies = st.sidebar.multiselect(
    "Selecciona tipo de genetica:",
    options=df["gen�tica"].unique(),
    default=df["gen�tica"].unique(),
)

anioSalida = st.sidebar.multiselect(
    "Selecciona por a�o de salida:",
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
    index=0,  # �ndice del valor por defecto (0 corresponde a 0)
)

peso_inicial = st.sidebar.selectbox(
    "Peso inicial: ",
    options=[20, 30, 40, 50],
    index=0,  # �ndice del valor por defecto (0 corresponde a 0)
)
peso_final = st.sidebar.selectbox(
    "Peso final: ",
    options=[220, 210, 230, 30],
    index=0,  # �ndice del valor por defecto (0 corresponde a 0)
)
poblacion = st.sidebar.selectbox(
    "Poblaci�n: ",
    options=[200000, 210000, 18000, 10],
    index=0,  # �ndice del valor por defecto (0 corresponde a 0)
)


# Aplicar filtros al DataFrame
filtro = (
    (df["gen�tica"].isin(especies))
    & (df["anioSalida"].isin(anioSalida))
    & (df["pesoInicial"] >= sepal_length[0])
    & (df["pesoInicial"] <= sepal_length[1])
)

df_filtrado = df[filtro]

# Mostrar tabla de datos
st.header("Datos de trabajo")
st.dataframe(df)

# Dise�o del Tablero
# Utilizar columnas para organizar el contenido
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuci�n de Conversi�n Alimenticia")

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

    # Agregar l�nea vertical en la media
    ax1.axvline(mean_conversion, color="red", linestyle="--", linewidth=2)

    # Agregar etiqueta para la media
    ax1.text(
        mean_conversion + 0.1,
        ax1.get_ylim()[1] * 0.9,
        f"Mean: {mean_conversion:.2f}",
        color="red",
        ha="center",
        fontsize=12,
        backgroundcolor="white",
    )

    # Configurar etiquetas y t�tulo
    ax1.set_xlabel("Conversi�n Alimenticia")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribuci�n de la Conversi�n Alimenticia con la Media")

    st.pyplot(fig1)
    st.markdown(
        """
            A nivel global (sin usar los filtros globales, la conversi�n alimenticia muestra una
            distribucci�n sesgada o inclinada hacia menores valores de conversi�n, con un valor
            medio en la muestra trabajada de  2.24 UA.
            """
    )

with col2:
    # st.header("Relaci�n Sepal vs. Petal")
    # fig2, ax2 = plt.subplots()
    # sns.scatterplot(data=df_filtrado, x='edadFinal',  y='conversionAlimenticia', ax=ax2)
    # st.pyplot(fig2)
    # Crear columnas internas para centrar la tabla

    st.subheader("Estad�sticas de Conversi�n Alimenticia")

    st.dataframe(df_filtrado.describe())
    st.markdown(
        """
            Dentro de las variables numericas podriamos inferir que la edad de los animales
            se mide en semanas, donde el proceso de medici�n inicia en promedio cuando los
            animales poseen 72 semanas  y concluye su vida en promedio a las 157 semanas. Estos
            animales en promedio empieza con un peso de 31 U.A y concluyen con un peso promedio de 113U.A
            La mortalidad tiene un promedio de 1.88% con un m�ximo de 10%.

            """
    )

# A�adir Box Plots
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>Box Plots: variables categ�ricas</h2>",
    unsafe_allow_html=True,
)
# Crear columnas para los box plots
box_col1, box_col2 = st.columns(2)

with box_col1:
    st.subheader("Conversi�n por fabrica de alimento")
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
    ax3.set_ylabel("Conversi�n")
    st.pyplot(fig3)
#
with box_col2:
    st.subheader("Conversi�n por tipo de gen�tica")
    fig4, ax4 = plt.subplots()
    sns.violinplot(
        x="gen�tica",
        y="conversionAlimenticia",
        data=df_filtrado,
        ax=ax4,
        palette="Set3",
        inner="quartile",
    )
    ax4.set_xlabel("Gen�tica")
    ax4.set_ylabel("Conversi�n")
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

    st.subheader("Conversi�n por los principales asesores")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="asesor", y="conversionAlimenticia", data=df_asseror, ax=ax3)
    ax3.set_xlabel("Asesor")
    ax3.set_ylabel("Conversi�n")
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

    st.subheader("Conversi�n por Granja")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="granja_anonym", y="conversionAlimenticia", data=df_granja, ax=ax4)
    ax4.set_xlabel("Granja")
    ax4.set_ylabel("Conversi�n")
    st.pyplot(fig4)

st.markdown(
    """
            - Las fabricas de alimento que mejor performance en conversi�n han tenido, es decir, el valor
            medio en conversi�n de la fabrica, comparado con el valor medio de toda la muestra de datos son pp_0
            y pp_8.  Caso opuesto se puede ver en la gen�tica de los animales, donde las categor�as lg_6 y lg_2 
            poseen un valor medio superior al de la muestra analizada.
            - En Adici�n, aunque en un mejor grado, los asesores rt_0 y rt_2 presentan un performance superior
            que los dem�s. En el caso de las granjas, se puede evidenciar un comportamiento algo atipico en el
            grupo de granjas para la gr_18 donde su valor medio degrada la conversi�n. De igual forma se observan 
            valores atipicos en la granja 48, pero estos valores mejoran el performance de conversi�n
            """
)
# Dispersi�n

# A�adir Box Plots
st.markdown(
    "<h2 style='text-align: center; color: #0D47A1;'>Relaci�n de disperci�n: variables num�ricas</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """En los siguientes graficos cada punto representa 
            un lote de cerdo y como la conversi�n alimenticia cambia en funci�n de
            alg�n predictor num�rico. Los puntos en rojo, son valores atipicos y estan calculados
            como aquellos datos que superan m�s de tres desviaciones est�ndar de la media.
            """
)

# TODO: scater
scatter_col1, scatter_col2 = st.columns(2)
df_filtrado["z_score_duracionCeba"] = stats.zscore(df_filtrado["duracionCeba"])

# Identificar outliers
outliers = np.abs(df_filtrado["z_score_duracionCeba"]) > 3
with scatter_col1:
    st.subheader("Dispersi�n de Duraci�n de Ceba y Conversi�n Alimenticia")
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
        label="Valores at�picos",
        alpha=0.8,
    )

    # Configuraci�n del gr�fico
    ax_scatter.set_title(
        "Dispersi�n de Duraci�n de Ceba y Conversi�n Alimenticia (con Valores At�picos)"
    )
    ax_scatter.set_xlabel("Duraci�n de Ceba (U.A)")
    ax_scatter.set_ylabel("Conversi�n Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """Lotes de cerdos con menos de 50 semanas de Ceba son casos anomalos, adiciones la conversi�n
    es superior al valor medio de la muestra total. A nivel descriptivo se observa
    que una conversi�n por debajo de la media se obtine con al menos 105 semanas de ceba, por encima de estas,
    la conversi�n se degrada
        """
    )

with scatter_col2:
    df_filtrado["z_score_edad_final"] = stats.zscore(df_filtrado["edadFinal"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_score_edad_final"]) > 3
    st.subheader("Dispersi�n de edad final y Conversi�n Alimenticia")

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
        label="Valores at�picos",
        alpha=0.8,
    )

    # Configuraci�n del gr�fico
    ax_scatter.set_title(
        "Dispersi�n de edadFinal y Conversi�n Alimenticia (con Valores At�picos)"
    )
    ax_scatter.set_xlabel("EdadFinal (meses)")
    ax_scatter.set_ylabel("Conversi�n Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
                La edad esta lineal mente relacionada con la conversi�n, por encima de 170 semanas
                de edad, los lotes de cerdos degradan el factor de conversi�n. Se evidencia tambien
                los valores atipicos en cerdos con edad final menor a 120 semanas los cuales para ser tan 
                jovenes degradaron su conversi�n seguro por algun factor externo.
            """
    )


scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    df_filtrado["z_score_mortalidad"] = stats.zscore(df_filtrado["mortalidad"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_score_mortalidad"]) > 3
    st.subheader("Dispersi�nmortalidad y Conversi�n Alimenticia")

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
        label="Valores at�picos",
        alpha=0.8,
    )

    # Configuraci�n del gr�fico
    ax_scatter.set_title(
        "Dispersi�n de mortalidad y Conversi�n Alimenticia (con Valores At�picos)"
    )
    ax_scatter.set_xlabel("mortalidad")
    ax_scatter.set_ylabel("Conversi�n Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
            En general existe baja mortalidad, pero aquellos lotes donde la mortalidad supera un 7%, 
            se tienen como anomal�as.
        """
    )

with scatter_col2:
    df_filtrado["z_pesoInicial"] = stats.zscore(df_filtrado["pesoInicial"])

    # Identificar outliers
    outliers = np.abs(df_filtrado["z_pesoInicial"]) > 3
    st.subheader("Dispersi�n pesoFinal y Conversi�n Alimenticia")

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
        label="Valores at�picos",
        alpha=0.8,
    )

    # Configuraci�n del gr�fico
    ax_scatter.set_title(
        "Dispersi�n pesoInicial y Conversi�n Alimenticia (con Valores At�picos)"
    )
    ax_scatter.set_xlabel("pesoInicial (U.A)")
    ax_scatter.set_ylabel("Conversi�n Alimenticia")
    ax_scatter.legend()

    st.pyplot(fig_scatter)
    st.markdown(
        """
                Lotes con pesos iniciales superior a 45 U.A se consideran anomal�as, adicional degradan
                la conversi�n.
            """
    )


st.markdown(
    """
                **Conclusiones generales a nivel descriptivo**.
                 - La conversi�n alimenticia tiene un promedio de 2.24 con un rango entre 1.80 y 2.79.
                 - El peso inicial promedio es 31.28 kg, y el peso final promedio es 113.90 kg.
                 - Los lotes de cerdo deberian de iniciar con un peso entre 25 y 40 Kg, para obtener al
                   final del proceso, conversiones por debajo del valor medio de la muestra analizada.
                 - El asesor rt_0 y rt_2 han mejorado el factor de conversi�n con respecto a los dem�s. 
                   Seria interezante revisar el volumne de lotes y cerdos en la cual se ha obtenido mejoras.
                 - Mortalidad tiene un promedio de 1.88% con un m�ximo de 10%.

    """
)
numerical_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_columns].corr()
st.header("=� Matriz de Correlaci�n")

st.markdown(
    """
Esta matriz de correlaci�n muestra la relaci�n entre las variables num�ricas del dataset. 
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
ax.set_title("Matriz de Correlaci�n entre Variables Num�ricas", fontsize=16, pad=20)
plt.tight_layout()

st.pyplot(fig)

st.markdown(
    """**Correlaciones**:
El mapa de calor muestra correlaciones entre variables Peso inicial y peso final tienen una correlaci�n
 positiva fuerte, lo cual es esperado.
Adicional y como es de esperar, el peso y edad inicial/final, 
respectivamente presentan correlacci�n positiva fuerte, es por tal motivo que en un modelo de machine learning
no es bueno tenerlas todas, para evitar colinealidades.
  Conversi�n alimenticia presenta relaciones moderadas con variables como mortalidad y peso inicial.
    """
)

# Secci�n de Modelado
st.header("=' Entrenamiento y Evaluaci�n del Modelo")

st.markdown(
    """
**Descripci�n del Modelo de Conversi�n Alimenticia:**  
En esta secci�n, entrenamos un modelo de regresi�n para predecir la conversi�n 
alimenticia basada en diversas caracter�sticas como la f�brica de alimento, granja,
 gen�tica, peso inicial y final, asesor�a y mortalidad. La conversi�n alimenticia
  es un indicador clave para evaluar la eficiencia alimentaria
   de los animales en la producci�n ganadera y el objetivo final es entender como y que valores
   o conjuntos de valores afectan esta conversi�n.
   
   Para esta tarea primero debemos transformar la data raw, aunque ya esta limpia, 
   debemos codificar las variables categ�ricas y normalizar las variables num�ricas para obtener
   un mejor performance en los modelos.
"""
)

# Funci�n para entrenar el modelo


st.header("=' Estructura del Pipeline de Preprocesamiento")
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
A continuaci�n se entrena diferentes modelos de regresi�n usando un AutoML, en este caso se uso pycaret y los
detalles se pueden observar en el notebook de trabajo, pero aca mostraremos las modelos entrenados
y los valores de las m�tricas obtenidas en el conjunto de test:
"""
)


@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_datos_modelos_entrenados():
    df = pd.read_csv("comparacion_modelos.csv")
    return df


# Mostrar tabla de datos
st.header("Comparaci�n de modelos entrenados")
st.dataframe(cargar_datos_modelos_entrenados())

st.markdown(
    """ **Cu�l modelo escoger?**

En base al valor de las m�tricas y la posterior interpretaci�n e importancia de los predictores
se escoje el Random Forest ya que reduce el valor de la MAE.
 
 MAE no penaliza excesivamente los errores grandes, lo que significa 
 que refleja de manera m�s equilibrada el rendimiento del modelo
  sin ser dominado por unos pocos outliers.
  
Los valores SHAP proporcionan una explicaci�n local (por instancia) y global (promedio), por lo que
 tener una m�trica como MAE facilita correlacionar la magnitud de 
 los errores con las contribuciones de las caracter�sticas.
 
Al entender que el MAE refleja el error promedio en
 la misma escala que los datos,
 puedes relacionar m�s directamente c�mo las 
 caracter�sticas influyen en este error promedio.
"""
)


st.header("Interpretraci�n de resultados en base a los Shap values")


@st.cache_data  # Utilizar cache_data para datos que no cambian
def cargar_shap_values():
    return joblib.load("shap_values.joblib")


shap_values = cargar_shap_values()


st.subheader("Gr�fico Beeswarm de SHAP")
fig7, ax7 = plt.subplots()
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig7)

st.markdown(
    """Para saber la importancia de las predicturas, usaremos los 
diagramas de SHAP ya que permite ver las contriciones de cada feature en el modelo
entrenado. Para este modelo la carateristica granja es la que contribuye m�s en la 
predici�n del modelo, donde valores altos en este par�metro dan un valor negativo
en el SHAP indicando una reducci�n en la variable predictora o conversi�n, lo que 
nos favorece para el negocio.
En esta misma feature, la concentracci�n de valores se tiene en un valor positivo 
de SHAP, indicando que en general valores menores en granja aumenta la conversi�n.
�Que significa un valor alto o peque�o en la granja? Depende de la transformaci�n 
de nuestros datos, un valor alto es que se tiene alta frecuencia, segun nuestra 
transformaci�n.

La mortalidad es otra feature facil de entender, valores peque�os en mortalidad 
reducen el SHAP y por tanto la conversi�n, como es de esperar y lo vimos en el 
an�lisis descriptivo.

En el caso de la genetica, vemos que valores grandes en la genetica, la categoria
que m�s se repite tiene a tener un valor negativo en SHAP, y por tanto reducir 
la conversi�n. Mientras que en general, valores peque�os en la genetica, aumentan
la conversi�n.

Si el peso inicial es peque�o, esto mejora al final la conversi�n. Algo opuesto
al peso final.

En el caso de el semestre de salida, no hay efecto si se saca en el primer
o segundo semestre la producci�n. Ya que la concentraci�n de puntos estan en 0.


Ahora, para ver que tanto contribuye una variable predictora en la predicci�n,
tendremos los siguientes gr�ficos tanto a nivel global como a nivel particular.
De destacar que los diferenciales en que contribuyen cada predictor son menores que 
el error absoluto (MAE).

"""
)

st.subheader("Gr�fico de barras de SHAP")
fig8, ax8 = plt.subplots()
shap.plots.bar(shap_values)
st.pyplot(fig8)


st.subheader("Gr�fico Waterfall de la Predicci�n Seleccionada")
index = st.number_input(
    "Ingrese el n�mero de la predicci�n (0 a N-1):",
    min_value=0,
    max_value=len(shap_values) - 1,
    step=1,
    value=0,
)

# Gr�fico Waterfall basado en la entrada del usuario
fig9, ax9 = plt.subplots()
shap.plots.waterfall(shap_values[int(index)], show=False)
st.pyplot(fig9)

# Gr�fico Force basado en la entrada del usuario
st.subheader("Gr�fico Force de la Predicci�n Seleccionada")
fig10 = shap.plots.force(shap_values[int(index)], matplotlib=True)
st.pyplot(fig10)
# st.subheader("Gr�fico Waterfall de la Primera Predicci�n")
# fig6, ax6 = plt.subplots()
# shap.plots.waterfall(shap_values[int(index)], show=False)
# st.pyplot(fig6)
st.markdown(
    """En las dos �ltimas graficas podemos ver el diferencial en contribuci�n que
hace que el valor medio de la predici�n sea diferente al valor medio de la poblaci�n.

"""
)


# Extraer los valores SHAP asociados a las columnas categ�ricas
categorical_columns = [
    col for col in cargar_procesados().columns if col in categorical_cols
]
# Extraer los valores SHAP asociados a las columnas categ�ricas
categorical_shap_values = pd.DataFrame(
    shap_values.values[
        :, [cargar_procesados().columns.get_loc(col) for col in categorical_columns]
    ],
    columns=categorical_columns,
)

# Calcular los valores m�nimos y m�ximos por variable
categorical_min = categorical_shap_values.min(axis=0)
categorical_max = categorical_shap_values.max(axis=0)

# Crear un DataFrame resumen para mostrar los resultados
categorical_summary = pd.DataFrame(
    {
        "Categor�a": categorical_min.index,
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

st.header("Aplicaci�n")
st.markdown(
    "5.	Suponga que el kilo de alimento cuesta $2.500 pesos."
    " En las siguientes variables categ�ricas: asesor,"
    " fabrica Alimento, granja y gen�tica, "
    "identifique el peor y el mejor nivel o categor�a"
    " seg�n su efecto marginal sobre la conversi�n."
    " Estime cu�nta diferencia de alimento se "
    "ahorra el mejor nivel respecto del peor sobre "
    "una poblaci�n de 200.000 cerdos  que pasa de 30"
    " a 220 kg de peso vivo y valorice dicha diferencia."
)
st.dataframe(categorical_summary)
# valor_kilo_alimento
# Footer
st.markdown("---")
st.markdown("**Hecho con d por [Juan David Rinc�n](jdrincone@gmail.com)**")
