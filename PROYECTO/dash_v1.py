import dash
import math
import pandas as pd
import plotly.graph_objs as go
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from pgmpy.inference import VariableElimination

#Estilo
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Importamos las librerías a usar
import pandas as pd
#Importamos los datos como dataframe
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv('processed_cleveland.data', names=columnas)

#Convertimos datos a números

#En donde tengamos datos faltantes (?) ponemos 0

for column in df.columns:
    # Reemplazar valores "?" con 0 en la columna y fila respectivas
    df[column] = df[column].replace('?', 0)

df = df.apply(pd.to_numeric, errors='coerce')    

#la variable age tiene que ser discreta para la red bayesiana, por lo que la dividiremos por cuartiles siendo 29 la edad minima y 77 la edad maxima 
df.loc[(df['age'] >= 29.0) & (df['age'] < 48.0), 'age_discreta'] = 1 
df.loc[(df['age'] >= 48.0) & (df['age'] < 56.0), 'age_discreta'] = 2 
df.loc[(df['age'] >= 56.0) & (df['age'] < 61.0), 'age_discreta'] = 3 
df.loc[(df['age'] >= 61.0) & (df['age'] <= 77.0), 'age_discreta'] = 4 
#la variable oldpeak tiene que ser discreta, por lo que la dividiremos por cuartiles siendo 0 el cuartil minimo y 6.2 el cuartil maximo
df.loc[(df['oldpeak'] >= 0) & (df['oldpeak'] < 0.800000), 'oldpeak_discreta'] = 1
df.loc[(df['oldpeak'] >= 0.800000) & (df['oldpeak'] < 1.600000), 'oldpeak_discreta'] = 2
df.loc[(df['oldpeak'] >= 1.600000) & (df['oldpeak'] <= 6.200000), 'oldpeak_discreta'] = 3 


#Variable num
df.loc[(df['num'] == 0) , 'num_discreta'] = 0
df.loc[(df['num'] != 0) , 'num_discreta'] = 1


#se crea la red bayesiana
from pgmpy.models import BayesianNetwork
model=BayesianNetwork(
    [("age_discreta","ca"),
     ("sex","thal"),
     ("thal","slope"),
     ("thal","exang"),
     ("slope","oldpeak_discreta"),
     ("oldpeak_discreta","ca"),
     ("exang","cp"),
     ("ca","num_discreta"),
     ("cp","num_discreta"),])

from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(
    data=df,
    estimator=MaximumLikelihoodEstimator
)
#for i in model.nodes():
 #   print(i)
  #  print(model.get_cpds(i))
    

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Crear variables para cada una de las variables
edad = dcc.Input(id='input-edad', type='number', placeholder='Inserte Edad')
sex = dcc.Input(id='input-sex', type='number', placeholder='Inserte Sexo')
thal = dcc.Input(id='input-thal', type='number', placeholder='Inserte Thal')
slope = dcc.Input(id='input-slope', type='number', placeholder='Inserte Slope')
oldpeak = dcc.Input(id='input-oldpeak', type='number', placeholder='Inserte Oldpeak')
exang = dcc.Input(id='input-exang', type='number', placeholder='Inserte Exang')
ca = dcc.Input(id='input-ca', type='number', placeholder='Inserte Ca')
cp = dcc.Input(id='input-cp', type='number', placeholder='Inserte Cp')


# Crear contenedor para el título
title_container = html.Div(
    children=[        html.Div(children=[            html.H1(children='Predictor de enfermedades cardíacas - Home Test', style={'text-align': 'center'})        ], style={'background-color': '#6495ED', 'padding': '10px', 'border-radius': '5px'})
    ]
)

subtitle_container = html.Div(
    children=[
        html.H2(children='Responda en cada una de las siguientes casillas con sus datos')
    ]
)

# Crear contenedor para los inputbox
input_container = html.Div(
    children=[
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Variable"),
                    html.Th("Valor"),
                    html.Th("Instrucciones")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Edad"),
                    html.Td(edad),
                    html.Td("Ingrese su edad en años")
                ]),
                html.Tr([
                    html.Td("Sexo"),
                    html.Td(sex),
                    html.Td("Ingrese su sexo formato binario (M=1/F=0)")
                ]),
                html.Tr([
                    html.Td("Thal"),
                    html.Td(thal),
                    html.Td("Ingrese el tipo de defecto talámico (3/6/7)")
                ]),
                html.Tr([
                    html.Td("Slope"),
                    html.Td(slope),
                    html.Td("Ingrese el tipo de la pendiente del segmento ST (0/1/2)")
                ]),
                html.Tr([
                    html.Td("Oldpeak"),
                    html.Td(oldpeak),
                    html.Td("Ingrese la depresión del segmento ST inducida por el ejercicio en relación con el reposo (0.0 a 6.2)")
                ]),
                html.Tr([
                    html.Td("Exang"),
                    html.Td(exang),
                    html.Td("Ingrese si hay presencia de angina inducida por ejercicio (0 = no, 1 = si)")
                ]),
                html.Tr([
                    html.Td("Ca"),
                    html.Td(ca),
                    html.Td("Ingrese el número de vasos principales coloreados por flourosopía (0/1/2/3)")
                ]),
                html.Tr([
                    html.Td("Cp"),
                    html.Td(cp),
                    html.Td("Ingrese el tipo de dolor torácico (0/1/2/3)")
                ])
            ])
        ])
    ]
)

# Crear contenedor para el botón
button_container = html.Div(
    children=[
        html.Br(),
        html.Button('Evaluar riesgo de enfermedad', id='boton_calcular', n_clicks=0),
        html.Br(), # Agregar espacio vacío
        html.Div(id="resultado") # Aquí se mostrará el resultado
    ]
)

# ---------------------------------------------
# Definir función para calcular la probabilidad
#----------------------------------------------

# Definir función para calcular la probabilidad
def calcular_probabilidad(age, sex, thal, slope, exang, oldpeak, ca, cp):
    if None in [age, sex, thal, slope, exang, oldpeak, ca, cp]:
        return "Por favor ingrese todos los valores"

    if age < 48:
        age_discreta = 1
    elif age < 56:
        age_discreta = 2   
    elif age < 61:
        age_discreta = 3   
    else:
        age_discreta = 4         

    if oldpeak < 0.8:
        oldpeak_discreta = 1   
    elif oldpeak < 1.6:
        oldpeak_discreta = 2
    else:
        oldpeak_discreta=3    
    #print(oldpeak_discreta)
    # Definir la evidencia para las variables
    evidence = {
                'age_discreta': age_discreta,
                'sex': sex,
                'thal': thal,
                'slope': slope,
                'exang': exang,
                'oldpeak_discreta': oldpeak_discreta,
                'ca': ca,
                'cp': cp
                }
    #print(evidence)
    infer = VariableElimination(model)

    # Calcular la probabilidad de la enfermedad cardíaca (num=1)
    q = infer.query(variables=['num_discreta'], evidence=evidence)
    
    #print(q)
    prob_enfermedad = q.values.tolist()[1]
    if math.isnan(prob_enfermedad):
        prob_enfermedad = 0
    #return prob_enfermedad
    #return prob_enfermedad #Bota proba de no enfermedad vs enfermedad [0,1]
    return html.P(f'La probabilidad de enfermedad cardíaca es: {prob_enfermedad[1]:.2f}')


# Agregar todo al layout
app.layout = html.Div(children=[
    title_container,
    subtitle_container,
    input_container,
    button_container
])


#Callbacks para conectar botón con función
@app.callback(
    Output('resultado', 'children'),
    Input('boton_calcular', 'n_clicks'),
    [Input('input-edad', 'value'),
     Input('input-sex', 'value'),
     Input('input-thal', 'value'),
     Input('input-slope', 'value'),
     Input('input-oldpeak', 'value'),
     Input('input-exang', 'value'),
     Input('input-ca', 'value'),
     Input('input-cp', 'value')]
)


def actualizar_output(n_clicks, input_edad, input_sex, input_thal, input_slope, input_oldpeak, input_exang, input_ca, input_cp):
    if n_clicks is not None:
        resultado = calcular_probabilidad(input_edad, input_sex, input_thal, input_slope, input_oldpeak, input_exang, input_ca, input_cp)
        return resultado

# Correr la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
