import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Open Sans'

#Parte a
pd.options.display.mpl_style = 'default'
data = pd.read_csv('data/data/titanic-train.csv',sep=';')

#Parte b

data.shape
data.info()
data.describe()


#Parte c
data.tail()
data.head()
data[100:110][:]
data[['Sex','Survived']].tail()
data[['Sex','Survived']][100:110]

#Parte d
data['Sex'].value_counts()
data.groupby('Sex').Survived.count()
data.groupby('Sex').Survived.mean()
data.groupby('Survived')['Sex'].value_counts()
grouped_props = data.groupby('Survived')['Sex'].value_counts()
data.groupby('Survived').size()
grouped_props.unstack().plot(kind='bar')

#Parte e
data.groupby('Survived')['Age'].mean()
data.boxplot(column='Age',by='Survived')
