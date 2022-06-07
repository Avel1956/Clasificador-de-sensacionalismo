from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import spacy
###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

###################################

from functionforDownloadButtons import download_button


###################################


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="🌀", page_title="Edición CSV")

# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)
st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/scissors_2702-fe0f.png",
    width=100,
)

st.title("Edición CSV")




c29, c30, c31 = st.columns([1, 6, 1])

with c30:

    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Verifique su archivo .csv")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Seleccione su archivo primero
                """
        )

        st.stop()

from st_aggrid import GridUpdateMode, DataReturnMode

gb = GridOptionsBuilder.from_dataframe(shows)
# enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
gridOptions = gb.build()

st.success(
    f"""
        💡 Mantenga pulsada la tecla SHIFT para seleccionar múltiples filas
        """
)

response = AgGrid(
    shows,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
)

df = pd.DataFrame(response["selected_rows"])

st.subheader("Filtered data will appear below 👇 ")
st.text("")

# st.table(df)

st.text("")

c29, c30, c31 = st.columns([1, 1, 2])

with c29:

    CSVButton = download_button(
        df,
        "File.csv",
        "Descargar como CSV",
    )

with c30:
    CSVButton = download_button(
        df,
        "File.csv",
        "Descargar como TXT",
    )
with c31:
    st.button("Aceptar")

if "Aceptar":
    c29, c30, c31 = st.columns([1, 6, 1])

with c30:
    st.subheader("Frecuencia de datos ")
    st.text("")

fig = px.histogram(df, x="labels")
st.plotly_chart(fig, use_container_width=True)
st.text("")

txt= st.text_area('Texto a clasificar', '''Este es un texto de muestra, reemplace por el suyo''')
def datransf(df, txt):
    # df["Palabras por texto"] = df["texto"].str.split().apply(len)
    

    nlp=spacy.load('es_core_news_sm')

    with nlp.disable_pipes():
        vectors_abstract = np.array([nlp(df.texto).vector for idx, df in df.iterrows()])


    # texto_a_comparar = str("ese personaje es una cucaracha que no merece nada mas que infierno") 
    with nlp.disable_pipes():
      abstract_test = np.array([nlp(txt).vector])
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(vectors_abstract, df.labels, 
                                                    test_size=0.5, random_state=3)
                                                        
    return abstract_test,X_train,X_test,y_train,y_test


abstract_test, X_train, X_test, y_train, y_test = datransf(df, txt)


def print_prediction(a):
  if a == 0:
    string1= ("El texto evaluado no es sensacionalista (class: "+ str(a))
    return(string1)
  else:
    string2= ("El texto evaluado es sensacionalista (class: "+ str(a)+")")
    return(string2)
################
# Modelos
################
################

################
# Lineal SVC
################
def SVC (df):

    from sklearn.svm import LinearSVC
    model = LinearSVC(random_state=1, dual=False, class_weight='balanced')
    model.fit(X_train, y_train)
    string = (f'LinearSVC accuracy: {model.score(X_test, y_test)*100:.3f}%')
    a= model.predict(abstract_test)
    return(string, a)

svc_res = list(SVC(df))
st.write("Resultados de algoritmo SVC:")
st.write(print_prediction(svc_res[0]))
st.write(print_prediction(svc_res[1]))



#################
#Naive Bayes
#################
def naive(df):

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="elasticnet", max_iter=30)
    clf.fit(X_train, y_train)
    string=(f'Naive Bayes accuracy: {clf.score(X_test, y_test)*100:.3f}%')
    a = clf.predict(abstract_test)
    return(string, a)
naive_res = list(naive(df))
st.write("Resultados de algoritmo Naive Bayes:")
st.write(print_prediction(naive_res[0]))
st.write(print_prediction(naive_res[1]))

#################
#K-Nearest
#################
def knear(df):    
    from sklearn.neighbors import NearestCentroid
    clf = NearestCentroid()
    clf.fit(X_train, y_train)
    string = (f'K-nearest accuracy: {clf.score(X_test, y_test)*100:.3f}%')
    a = clf.predict(abstract_test)
    return(string, a)
knear_res = list(knear(df))
st.write("Resultados de algoritmo K-Nearest:")
st.write(print_prediction(knear_res[0]))
st.write(print_prediction(knear_res[1]))
#################
#Gaussian Naive Bayes
#################
def gausnaive(df):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    string = (f'Gaussian Naive Bayes accuracy: {gnb.score(X_test, y_test)*100:.3f}%')
    a = gnb.predict(abstract_test)
    return(string, a)
gausnaive_res = list(gausnaive(df))
st.write("Resultados de algoritmo Gaussian Naive Bayes:")
st.write(print_prediction(gausnaive_res[0]))
st.write(print_prediction(gausnaive_res[1]))
#################
#Decision tree
#################
def dectree(df):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    string = (f'Decision tree accuracy: {clf.score(X_test, y_test)*100:.3f}%')
    a = clf.predict(abstract_test)
    return(string, a)
dectree_res = list(dectree(df))
st.write("Resultados de algoritmo Arbol de decisión:")
st.write(print_prediction(dectree_res[0]))
st.write(print_prediction(dectree_res[1]))


#################
#Randomized Decision tree
#################
def rdectree(df):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)
    string = (f'Randomized decision tree accuracy: {clf.score(X_test, y_test)*100:.3f}%')
    a = clf.predict(abstract_test)
    return(string, a)
rdectree_res = list(rdectree(df))
st.write("Resultados de algoritmo Gaussian Naive Bayes:")
st.write(print_prediction(rdectree_res[0]))
st.write(print_prediction(rdectree_res[1]))

 
##################
#Multilayer perceptron
#################
def percep(df):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    string = (f'Multi-layer perceptron accuracy: {clf.score(X_test, y_test)*100:.3f}%')
    a = clf.predict(abstract_test)
    return(string, a)
percep_res = list(percep(df))
st.write("Resultados de algoritmo Gaussian Naive Bayes:")
st.write(print_prediction(percep_res[0]))
st.write(print_prediction(percep_res[1]))

 