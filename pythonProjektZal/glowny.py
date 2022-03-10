import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# WCZYTANIE BAZY DANYCH ORAZ PRZYPISANIE ROŚLIN DO ODPOWIEDNICH GATUNKÓW

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])
df['target'] = pd.Series(iris.target)
df['target names'] = pd.Series(iris.target_names)
gatunki = []
for i in range(len(df)):
    if df.iloc[i]['target'] == 0:
        gatunki.append('setosa')
    elif df.iloc[i]['target'] == 1:
        gatunki.append('versicolor')
    elif df.iloc[i]['target'] == 2:
        gatunki.append('virginica')
df['Gatunki'] = gatunki

x = iris.data

# WIZUALIZACJA DANYCH NA WYKRESIE PRZED SORTOWANIEM METODĄ K-MEANS

plt.scatter(x=df['sepal-length'], y=df['sepal-width'], c=iris.target, cmap='gist_rainbow')
plt.xlabel('Sepal width', fontsize=18)
plt.ylabel('Sepal length', fontsize=18)

plt.show()
"""
# WYSZUKIWANIE ODPOWIEDNIEGO k ZA POMOCĄ METODY "ELBOW" i WCSS, PRZYJMUJEMY LICZBE KLASTRÓW OD 1 DO 11

WCSS = []
for i in range(1, 11):
    kmeans11 = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(x)
    kmeans11.fit(x)
    WCSS.append(kmeans11.inertia_)


plt.plot(range(1, 11), WCSS)
plt.title('Metoda łokcia dla k 1-11')
plt.xlabel('Liczba klastrów')
plt.ylabel('WCSS')
plt.show()
"""

# SORTOWANIE METODĄ K-MEANS (WBUDOWANĄ W BIBLIOTEKĘ SKLEARN)

kmeans3 = KMeans(n_clusters=3, random_state=2)
y = kmeans3.fit_predict(x)

plt.scatter(x[y == 0, 0], x[y == 0, 1], s=15, c='cyan', label='Klaster_1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=15, c='blue', label='Klaster_2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=15, c='green', label='Klaster_3')
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s=25, c='red', label='Centroidy')
plt.title('Przewidywany podział algorytmu K-means - 2 atrybuty')
plt.legend()
plt.show()



