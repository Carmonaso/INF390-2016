import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.preprocessing import StandardScaler
import math
"""
#Parte  (a)
#Se limpia el dataframe, se eliminan los anos 1979 al 1989
#Se eliminan los valores nulos del  ano 2010
data = pd.read_csv('data/data/HIV.csv',sep=';',decimal=',', index_col = 0)
data = data.drop(data.columns[range(0,11)], axis=1)
data = data[data['2010'].notnull()]
data = data.dropna()
data.shape
data.index.names = ['country']
data.columns.names = ['year']


#Parte (b)
#Grafico que muestra evolucion del VIH en distintos paises
fig, ax = plt.subplots(figsize=(8, 4))
data.loc[['Chile','Argentina','Italy','Cameroon', 'Congo, Rep,'],'1990':].T.plot(ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1),prop={'size':'x-small'},ncol=6)
plt.tight_layout(pad=1.5)
#plt.show()

#Parte (c)
X = data.ix[:,'1990':'2011'].values
X_std = StandardScaler().fit_transform(X)


#Parte (d) PCA

#Realizado  con EIG
cov_mat = np.transpose(X_std).dot(X_std) / (X_std.shape[0]-1)
valores, vectores = np.linalg.eig(cov_mat)
#print('Eigenvectors EIG\n%s' %vectores)
#print('\nEigenvalues EIG\n%s' %valores)


#Realizado con SVD
U,s,V =np.linalg.svd(cov_mat)
eig_pairs = [(np.abs(s[i]), U[:,i]) for i in range(len(valores))]
eig_pairs.sort()
eig_pairs.reverse()

#Grafico  varianzas
tot = sum(valores)
var_exp = [(i / tot)*100 for i in sorted(valores, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(6, 4))
var_exp=var_exp[:4]
cum_var_exp=cum_var_exp[:4]
plt.bar(range(4), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()


#Proyeccion  matriz
matrix_w = np.hstack((eig_pairs[0][1].reshape(22,1),
                      eig_pairs[1][1].reshape(22,1)))


Y = X_std.dot(matrix_w)


data_2d = pd.DataFrame(Y)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']

#Parte (e)
#Plot Scatter por promedio
row_means = tuple(data.mean(axis=1))
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_means,cmap='Reds')


#Plot Scatter por diferencia entre los anos
row_trends=tuple(data.diff(1).mean(axis=1))
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_trends,cmap='RdBu')

#Parte (f)
#Scatters Modificado
#data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), s=10000*row_means,c=row_means)
#data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), s=10000*row_trends,c=row_trends)


#Parte(g)
fig, ax = plt.subplots(figsize=(16,8))
row_means = tuple(data.mean(axis=1))
row_trends = tuple(data.diff(1).mean(axis=1))
print row_means
data_2d.plot(kind='scatter', x='PC2', y='PC1', ax=ax, s=10*row_means, c=row_means, cmap='RdBu')
Q3_HIV_world = data.mean(axis=1).quantile(q=0.85)
HIV_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
    if(HIV_country[i]>Q3_HIV_world):
        ax.annotate(txt, (data_2d.iloc[i].PC2+0.2,data_2d.iloc[i].PC1))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

"""


#######################################
########### Tuberculosis ##############
#######################################
data = pd.read_csv('data/data/TB.csv',sep=',',thousands=',', index_col = 0)
data.index.names = ['country']
data.columns.names = ['year']
X = data.ix[:,'1990':'2007'].values
X_std = StandardScaler().fit_transform(X)

#Parte (d)
cov_mat = np.transpose(X_std).dot(X_std) / (X_std.shape[0]-1)
valores, vectores = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(valores[i]), vectores[:,i]) for i in range(len(valores))]
eig_pairs.sort()
eig_pairs.reverse()

#Grafico  varianzas
tot = sum(valores)
var_exp = [(i / tot)*100 for i in sorted(valores, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(6, 4))
var_exp=var_exp[:4]
cum_var_exp=cum_var_exp[:4]
plt.bar(range(4), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Proyeccion  matriz
matrix_w = np.hstack((eig_pairs[0][1].reshape(18,1),
                      eig_pairs[1][1].reshape(18,1)))
Y = X_std.dot(matrix_w)

data_2d = pd.DataFrame(Y)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']

#Parte (e)
#Plot Scatter por promedio
row_means = tuple(data.mean(axis=1))
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_means,cmap='Reds')


#Plot Scatter por diferencia entre los anos
row_trends=tuple(data.diff(1).mean(axis=1))
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_trends,cmap='RdBu')

#Parte (f)
#Scatters Modificado
#data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), s=10000*row_means,c=row_means)
#data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), s=10000*row_trends,c=row_trends)


#Parte(g)
fig, ax = plt.subplots(figsize=(16,8))
row_means = tuple(data.mean(axis=1))
row_trends = tuple(data.diff(1).mean(axis=1))
data_2d.plot(kind='scatter', x='PC2', y='PC1', ax=ax, s=10*row_means, c=row_means, cmap='RdBu')
Q3_HIV_world = data.mean(axis=1).quantile(q=0.85)
HIV_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
    if(HIV_country[i]>Q3_HIV_world):
        ax.annotate(txt, (data_2d.iloc[i].PC2+0.2,data_2d.iloc[i].PC1))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
