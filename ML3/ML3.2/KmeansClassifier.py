import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA



class KmeansClassifier:
    def __init__(self, k=3, max_iter=100, X=None):
        self.k = k
        self.max_iter = max_iter

    def __str__(self) -> str:
        return f"KmeansClassifier(k={self.k}, max_iter={self.max_iter})"

    def show_data(self):
        # print(self.X)
        return self.X

    def check_data(self):
        if not self.X:
            raise Exception(
                "X is not defined - please use .load_data(X) to load data before")

    def load_data(self, X):
        if X is not None:
            self.X = X.data
            self.target = X.target
        else:
            raise Exception("X cannot be None")

    def init_centroids(self):

        centroids = self.X[np.random.choice(
            self.X.shape[0], self.k, replace=False)]

        self.centroids = centroids

        return centroids

    # Cette fonction attend un dataframe de la forme IRIS pour avoir le .data et .target

    def plot(self):

        if self.X is not None:
            if self.centroids is not None:
                self.init_centroids()

            pca = PCA(n_components=2)

            X_pca = pca.fit_transform(self.X)
            centroids_pca = pca.transform(self.centroids)

            df_X = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_centroids = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])

            # df_pca['label'] = self.target

            # df_pca['label'] = df_pca['label'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                # hue='label',
                data=df_X,
                palette='Set2',
                marker='x',
                label="Data"
            )

            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                # hue='label',
                data=df_centroids,
                palette='tab10',
                s=100,
                marker='o',
                alpha=.5,
                label="Centroids"
            )

            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('PCA on Iris Dataset')
            plt.show()
            # time.sleep(1)
            # plt.close()

            
        else:
            raise Exception("X cannot be None")
            