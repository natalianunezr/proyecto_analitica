import dash
import pandas as pd
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from pgmpy.inference import VariableElimination


#Importamos los datos como dataframe
df = pd.read_csv('processed.cleveland.csv', header=0)

#Convertimos datos a números
df = df.apply(pd.to_numeric, errors='coerce')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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

# Crear contenedores para el título y subtítulo
title_container = html.Div(
    children=[
        html.H1(children='Predictor de enfermedades cardíacas - Home Test')
    ]
)

#Agrego logo
image_container = html.Div(
    children=[
        html.Img(src= 'C:/Users/natty/proyectoanalitica/logo_uniandes.png ', style={'width': '500px'})
    ],
    style={'display': 'flex', 'justify-content': 'flex-end'}
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
                    html.Td("Ingrese el tipo de defecto talámico (0/1/2/3)")
                ]),
                html.Tr([
                    html.Td("Slope"),
                    html.Td(slope),
                    html.Td("Ingrese el tipo de la pendiente del segmento ST (0/1/2)")
                ]),
                html.Tr([
                    html.Td("Oldpeak"),
                    html.Td(oldpeak),
                    html.Td("Ingrese la depresión del segmento ST inducida por el ejercicio en relación con el reposo")
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
        html.Br() # Agregar espacio vacío
    ]
)

@app.callback(Output('contenedor_resultados', 'children'),
              [Input('boton_calcular', 'n_clicks')],
              [State('input-edad', 'value'),
               State('input-sex', 'value'),
               State('input-thal', 'value'),
               State('input-slope', 'value'),
               State('input-exang', 'value'),
               State('input-oldpeak', 'value'),
               State('input-ca', 'value'),
               State('input-cp', 'value')])

# Definir función para calcular la probabilidad
def calcular_probabilidad(age, sex, thal, slope, exang, oldpeak, ca, cp):
    

    if age >= 29.0 and age < 48:
        age_discreta = 1
    if age >= 48 and age < 56:
        age_discreta = 2   
    if age >= 56 and age < 61:
        age_discreta = 3   
    if age >= 61 and age <= 77:
        age_discreta = 4         

    
    # Cargar la red bayesiana y crear el objeto de inferencia
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model = BayesianNetwork(
        [("age_discreta","ca"),
         ("sex","thal"),
         ("thal","slope"),
         ("thal","exang"),
         ("slope","oldpeak_discreta"),
         ("oldpeak_discreta","ca"),
         ("exang","cp"),
         ("ca","num_discreta"),
         ("cp","num_discreta"),])
    model.fit(data=df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)

    # Definir la evidencia para las variables
    evidence = {'age_discreta': age_discreta,
                'sex': sex,
                'thal': thal,
                'slope': slope,
                'exang': exang,
                'oldpeak_discreta': oldpeak,
                'ca': ca,
                'cp': cp}

    # Calcular la probabilidad de la enfermedad cardíaca (num=1)
    q = infer.query(variables=['num_discreta'], evidence=evidence)
    #print(q)
    prob_enfermedad = q.values.tolist()
    #return prob_enfermedad #Bota proba de no enfermedad vs enfermedad [0,1]
    return html.P(f'La probabilidad de enfermedad cardíaca es: {prob_enfermedad[1]:.2f}')

def actualizar_resultados(edad, sex, thal, slope, exang, oldpeak, ca, cp):
    prob_enfermedad = calcular_probabilidad(
        float(edad.value),
        int(sex.value),
        int(thal.value),        #ERROR EN TODAS, NO SE PUEDE USAR .VALUE CON OBJETO INPUT 
        int(slope.value),
        int(exang.value),
        float(oldpeak.value),
        int(ca.value),
        int(cp.value)
    )
    return prob_enfermedad

prob_enfermedad = actualizar_resultados(edad, sex, thal, slope, exang, oldpeak, ca, cp)
resultados_finales = html.Div(id='contenedor_resultados', children=[prob_enfermedad])

# Agregar todo al layout
app.layout = html.Div(children=[
    title_container,
    image_container,
    subtitle_container,
    input_container,
    button_container,
    resultados_finales
])

# Correr la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
