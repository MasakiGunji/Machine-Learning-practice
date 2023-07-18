import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

iris_data = iris.data

st_data = iris_data[:50]
vc_data = iris_data[50:100]
plt.scatter(st_data[:,0], st_data[:,1], label="Setosa")
plt.scatter(vc_data[:,0], vc_data[:,1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show