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
     ("ca","num"),
     ("cp","num"),])
from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(
    data=df,
    estimator=MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(i)
    print(model.get_cpds(i))

# Definir función para calcular la probabilidad
from pgmpy.inference import VariableElimination
def calcular_probabilidad(age, sex, thal, slope, exang, oldpeak, ca, cp):
    # Cargar la red bayesiana y crear el objeto de inferencia
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
print(calcular_probabilidad(61,1,6,3,1,2,2,4))

