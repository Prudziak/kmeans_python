import glowny as gl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

x = gl.iris.data
a = (0.0, 0.0, 0.95, 1.0)
est = gl.KMeans(n_clusters=3, random_state=2)
fignum = 1
y = 0

# WIZUALIZACJA PRZED KLASTROWANIEM - 3 ATRYBUTY

fig1 = plt.figure(fignum, figsize=(4, 3))
ax1 = Axes3D(fig1, rect=a, elev=48, azim=134, auto_add_to_figure=False)
fig1.add_axes(ax1)

ax1.scatter(x[:, 3], x[:, 0], x[:, 2], c=gl.iris.target)
ax1.w_xaxis.set_ticklabels([])
ax1.w_yaxis.set_ticklabels([])
ax1.w_zaxis.set_ticklabels([])
ax1.set_xlabel("Petal width")
ax1.set_ylabel("Sepal length")
ax1.set_zlabel("Petal length")
ax1.set_title("Wizualizacja dla 3 atrybutów przed sortowaniem")
ax1.dist = 12

plt.show()

# teraz sortowanie przy uzyciu metody k-srednich
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=a, elev=48, azim=134, auto_add_to_figure=False)
fig.add_axes(ax)
est.fit(x)
labels = est.labels_

#WIZUALIZACJA PO KLASTROWANIU METODĄ K-MEANS - 3 ATRYBUTY

ax.scatter(x[:, 3], x[:, 0], x[:, 2], c=labels.astype(float))
ax.scatter(est.cluster_centers_[:, 3], est.cluster_centers_[:, 0], est.cluster_centers_[:, 2], s=25, c='red',
           label='Centroidy')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.set_title("Wizualizacja podziału dla 3 atrybutów")
ax.dist = 12

plt.legend()
plt.show()

centroids = est.cluster_centers_
# print(centroids)

# dokładność klastrowania dla 4 atrybutów
score = metrics.accuracy_score(gl.df['target'], labels)
# print(score)

