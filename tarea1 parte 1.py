#Importaciones necesarias
import pandas as pd

#Cargar datos de entrada
pd.options.display.mpl_style = 'default'
data = pd.read_csv('titanic-train.csv', sep=';')

#Cantidad de items y variables
cantidad_items, cantidad_variables = data.shape

#Obtener nombres de variables
nombres_variables = list(data.columns.values)

#Tipo de cada variable, asi como cantidad de valores no nulos
data.info()

#Informacion estadistica de cada variable
data.describe()

#Se muestran los primeros 4 items del dataframe
primeros = data.head()
#Se muestran los ultimos 4 items del dataframe
ultimos = data.tail()
#Se muestran los ítemes 200 al 210
itemes_200_210 = data[199:210][:]
#Se repiten los tres pasos anteriores, pero observando solo las variables Sex y Survived
primeros_sex_survived = data[['Sex','Survived']].head()
ultimos_sex_survived = data[['Sex','Survived']].tail()
itemes_200_210_sex_survived = data[['Sex','Survived']][199:210]

#Determinar proporcion de hombres y mujeres que sobrevivieron
hombres = data[data.Sex == 'male']
proporcion_hombres_sobrevivientes = float(sum(hombres.Survived))/len(hombres)
mujeres = data[data.Sex == 'female']
proporcion_mujeres_sobrevivientes = float(sum(mujeres.Survived))/len(mujeres)

#Graficar proporcion de hombres y mujeres que sobrevivieron
data['Survived'] = data['Survived'].replace(1, 'Did survive')
data['Survived'] = data['Survived'].replace(0, 'Did not survive')
proporciones1 = data.groupby('Sex')['Survived'].value_counts() / \
data.groupby('Sex').size()
grafico1 = proporciones1.unstack().plot(kind='bar').set_ylabel('Proportions')

#Determinar proporcion de sobrevivientes que son hombres y mujeres
sobrevivientes = data[data.Survived == 1]
proporcion_sobrevivientes_hombres = float(sum(sobrevivientes.Sex == 'male'))/len(sobrevivientes)
proporcion_sobrevivientes_mujeres = float(sum(sobrevivientes.Sex == 'female'))/len(sobrevivientes)

#Graficar proporcion de sobrevivientes que son hombres y mujeres
proporciones2 = data.groupby('Survived')['Sex'].value_counts() / \
data.groupby('Survived').size()
grafico2 = proporciones2.unstack().plot(kind='bar')

#Edad media de pasajeros que sobreviveron
edad_media_sobrevivientes = data.groupby('Survived')['Age'].mean()
#Obtener boxplot de la categoria anterior
boxplot1 = data.boxplot(column='Age', by='Survived')
#Obtener histograma de la categoria anterior
hist1 = data.hist(column='Age', by='Survived')

#Obtener datos del pasajero que posee la edad maxima
datos_edad_maxima = data[data['Age'] == data['Age'].max()]

#Imputar edades faltantes, usando promedio de edades de cada sexo
promedio_hombres = data[(data['Age'].notnull()) & (data['Sex'] == 'male')]['Age'].mean()
promedio_mujeres = data[(data['Age'].notnull()) & (data['Sex'] == 'female')]['Age'].mean()
data.loc[(data['Age'].isnull()) & (data['Sex'] == 'male'), 'Age'] = promedio_hombres
data.loc[(data['Age'].isnull()) & (data['Sex'] == 'female'), 'Age'] = promedio_mujeres

#Determinar cantidad de clases diferentes de pasajeros
cantidad_clases = data['Pclass'].unique()

#Proporcion de sobrevivientes por cada clase
clase1 = data[data.Pclass == 1]
proporcion_clase1_sobrevivientes = float(sum(clase1.Survived))/len(clase1)
clase2 = data[data.Pclass == 2]
proporcion_clase2_sobrevivientes = float(sum(clase2.Survived))/len(clase2)
clase3 = data[data.Pclass == 3]
proporcion_clase3_sobrevivientes = float(sum(clase3.Survived))/len(clase3)
#Construir grafico para mostrar la informacion anterior
proporciones3 = data.groupby('Pclass')['Survived'].value_counts() / \
data.groupby('Pclass').size()
grafico3 = proporciones3.unstack().plot(kind='bar').set_ylabel('Proportions')

#Proporcion de hombres sobrevivientes por cada clase
hombres_sobrevivientes_xclase = data[data.Sex == 'male'].groupby(['Survived', 'Pclass']).size()/ \
data[data.Sex == 'male'].groupby(['Survived']).size()
hombres_sobrevivientes_xclase.unstack().plot(kind='bar', title='Males').set_ylabel('Proportions')

#Proporcion de mujeres sobrevivientes por cada clase
mujeres_sobrevivientes_xclase = data[data.Sex == 'female'].groupby(['Survived', 'Pclass']).size()/ \
data[data.Sex == 'female'].groupby(['Survived']).size()
mujeres_sobrevivientes_xclase.unstack().plot(kind='bar', title='Females').set_ylabel('Proportions')

#Construccion de regla de prediccion de sobrevivencia en base a sexo y clase de pasajero
#Se construye nueva columna
data['prediction'] = 0
#Se predicen los sobrevivientes
data.loc[(data.Sex == 'female') & (data.Pclass == 1), 'prediction'] = 1
nuevo_orden = [11] + range(11)
data.reindex(columns=data.columns[nuevo_orden])
#Se calcula precision para sobrevivientes
precision1 = data[data.prediction == 1][data.Survived == 'Did survive'].size / float(data[data.prediction == 1].size)
#Se calcula precision para no sobrevivientes
precision2 = data[data.prediction == 0][data.Survived == 'Did not survive'].size / float(data[data.prediction == 0].size)
#Se calcula recall para sobrevivientes
recall1 = data[data.prediction == 1][data.Survived == 'Did survive'].size / float(data[data.Survived == 'Did survive'].size)
#Se calcula recall para no sobrevivientes
recall2 = data[data.prediction == 0][data.Survived == 'Did not survive'].size / float(data[data.Survived == 'Did not survive'].size)
#Guardar archivo con nueva columna
data.to_csv('predicciones-titanic.csv', index=False)

#Boxplot para precio del boleto (variable Fare)
data.boxplot(column='Fare', by='Survived')
#Filtrar outliers
data_filt = data[data.Fare < 52]
#Histograma de data_filt
hist2 = data_filt.hist(column='Fare', by='Survived')

#Creacion de variable rango de precio
data['rangoPrecio'] = ''
#Se asigna un rango para la variable creada en cada ítem
data.loc[(data.Fare >= 0) & (data.Fare <= 9), 'rangoPrecio'] = '[0,9]'
data.loc[(data.Fare >= 10) & (data.Fare <= 19), 'rangoPrecio'] = '[10,19]'
data.loc[(data.Fare >= 20) & (data.Fare <= 29), 'rangoPrecio'] = '[20,29]'
data.loc[(data.Fare >= 29) & (data.Fare <= 39), 'rangoPrecio'] = '[29,39]'
data.loc[data.Fare >= 40, 'rangoPrecio'] = '[40+]'
nuevo_orden = [12] + range(12)
data.reindex(columns=data.columns[nuevo_orden])

#Se abre conjunto de datos de entrenamiento
datatest = pd.read_csv('/home/Seba/AID/Tarea1/titanic-test.csv', sep=',')
d = pd.read_csv('/home/Seba/AID/Tarea1/gendermodel.csv', sep=',')
columna_survived = d['Survived']
datatest['Survived'] = 0
datatest['Survived'] = columna_survived
#Se agrega columna predictiva
datatest['prediction'] = 0
#Se predicen los sobrevivientes
datatest.loc[(datatest.Sex == 'female') & (datatest.Pclass == 1), 'prediction'] = 1

nuevo_orden = [11] + range(11)
datatest.reindex(columns=data.columns[nuevo_orden])

#Se calcula precision y recall sobre conjunto de datos de entrenamiento
#Se calcula precision para sobrevivientes
precision3 = datatest[datatest.prediction == 1][datatest.Survived == 1].size / float(datatest[datatest.prediction == 1].size)
#Se calcula precision para no sobrevivientes
precision4 = datatest[datatest.prediction == 0][datatest.Survived == 0].size / float(datatest[datatest.prediction == 0].size)
#Se calcula recall para sobrevivientes
recall3 = datatest[datatest.prediction == 1][datatest.Survived == 1].size / float(datatest[datatest.Survived == 1].size)
#Se calcula recall para no sobrevivientes
recall4 = datatest[datatest.prediction == 0][datatest.Survived == 0].size / float(datatest[datatest.Survived == 0].size)
#Guardar archivo con nueva columna
data.to_csv('predicciones-test-titanic.csv', index=False)



















