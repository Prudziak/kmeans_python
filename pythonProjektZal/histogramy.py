import matplotlib.pyplot as plt

import glowny as gl

# Histogram 1 - sepal length

plt.figure(figsize=(10, 7))
x = gl.df["sepal-length"]
plt.hist(x, bins=20, color='blue')
plt.title("Histogram - sepal length")
plt.xlabel("Dlugosc (cm)")
plt.ylabel("Ilosc")

plt.show()

# Histogram 2 - petal length

plt.figure(figsize=(10, 7))
y = gl.df["petal-length"]
plt.hist(y, bins=20, color='green')
plt.title("Histogram - petal length")
plt.xlabel("Dlugosc (cm)")
plt.ylabel("Ilosc")

plt.show()

# Histogram 3 - sepal width

plt.figure(figsize=(10, 7))
z = gl.df["sepal-width"]
plt.hist(z, bins=20, color='purple')
plt.title("Histogram - sepal width")
plt.xlabel("Szerokosc (cm)")
plt.ylabel("Ilosc")

plt.show()