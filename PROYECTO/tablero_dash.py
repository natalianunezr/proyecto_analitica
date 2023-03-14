import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from pgmpy.inference import VariableElimination


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children="Predictor de enfermedades cardíacas - Home Test"),
    html.H2(children= "Responda en cada una de las siguientes casillas con sus datos"),
    html.Div([
        html.Div(["Edad: ", dcc.Input(id="input-age", type="number", value=18)]),
        html.Div(["Sexo: ", dcc.RadioItems(
            id="input-sex",
            options=[
                {'label': 'Masculino', 'value': 'M'},
                {'label': 'Femenino', 'value': 'F'}
            ],
            value='M'
        )]),
        html.Div(["Thal: ", dcc.Dropdown(
            id="input-thal",
            options=[
                {'label': 'Normal', 'value': 'Normal'},
                {'label': 'Defectuoso', 'value': 'Defectuoso'},
                {'label': 'Reversible', 'value': 'Reversible'}
            ],
            value='Normal'
        )]),
        html.Div(["Slope: ", dcc.Dropdown(
            id="input-slope",
            options=[
                {'label': 'Up', 'value': 'Up'},
                {'label': 'Flat', 'value': 'Flat'},
                {'label': 'Down', 'value': 'Down'}
            ],
            value='Up'
        )]),
        html.Div(["Exang: ", dcc.RadioItems(
            id="input-exang",
            options=[
                {'label': 'Si', 'value': 'Si'},
                {'label': 'No', 'value': 'No'}
            ],
            value='Si'
        )]),
        html.Div(["Oldpeak: ", dcc.Input(id="input-oldpeak", type="number", value=0.0)]),
        html.Div(["Ca: ", dcc.Input(id="input-ca", type="number", value=0)]),
        html.Div(["Cp: ", dcc.Dropdown(
            id="input-cp",
            options=[
                {'label': 'Angina típica', 'value': 'Typical'},
                {'label': 'Angina atípica', 'value': 'Atypical'},
                {'label': 'Dolor no anginoso', 'value': 'Non-anginal'},
                {'label': 'Asintomático', 'value': 'Asymptomatic'}
            ],
            value='Typical'
        )])
    ]),
    html.Div(id="output-div")
])
# Definir función para calcular la probabilidad
def calcular_probabilidad(age, sex, thal, slope, exang, oldpeak, ca, cp):
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
         ("ca","num"),
         ("cp","num"),])
    model.fit(data=df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)

    # Definir la evidencia para las variables
    evidence = {'age_discreta': age,
                'sex': sex,
                'thal': thal,
                'slope': slope,
                'exang': exang,
                'oldpeak_discreta': oldpeak,
                'ca': ca,
                'cp': cp}

    # Calcular la probabilidad de la enfermedad cardíaca (num=1)
    q = infer.query(variables=['num'], evidence=evidence)
    prob_enfermedad = q['num'].values[1]
    return prob_enfermedad

# Agregar la función al callback
@app.callback(Output(component_id="output-div", component_property="children"), 
              [Input(component_id="input-age", component_property="value"),
               Input(component_id="input-sex", component_property="value"),
               Input(component_id="input-thal", component_property="value"),
               Input(component_id="input-slope", component_property="value"),
               Input(component_id="input-exang", component_property="value"),
               Input(component_id="input-oldpeak", component_property="value"),
               Input(component_id="input-ca", component_property="value"),
               Input(component_id="input-cp", component_property="value")])
def update_output_div(age, sex, thal, slope, exang, oldpeak, ca, cp):
    # Calcular la probabilidad
    prob_enfermedad = calcular_probabilidad(age, sex, thal, slope, exang, oldpeak, ca, cp)

    # Crear mensaje con el resultado
    if prob_enfermedad > 0.5:
        mensaje = "El resultado sugiere que hay una alta probabilidad de que usted tenga una enfermedad cardíaca."
    else:
        mensaje = "El resultado sugiere que hay una baja probabilidad de que usted tenga una enfermedad cardíaca."
    return mensaje
