# We are going to implement a kmeans algorithm
# We will code each function separately and then put them together
# we will build a class to hold the functions
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from KmeansClassifier import KmeansClassifier
from sklearn.decomposition import PCA

iris = load_iris()

clf = KmeansClassifier()
pca = PCA(n_components=2)

X_pca = pca.fit_transform(iris.data)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

df_pca['target'] = iris.target

df_pca['target'] = df_pca['target'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(
    ax=ax,
    x='PC1', 
    y='PC2', 
    hue='target', 
    data=df_pca, 
    palette='tab10'
    )

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA on Iris Dataset')

fig.show()

