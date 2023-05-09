import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from numpy.linalg import norm
import matplotlib.animation as animation


class KmeansClassifier:
    """
    
    """
    def __init__(self, k=3, max_iter=100, X=None, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.history = {}

    def __str__(self) -> str:
        return f"KmeansClassifier(k={self.k}, max_iter={self.max_iter})"

    def check_data(self):
        if not self.X:
            raise Exception(
                "X is not defined - please use .load_data(X) to load data before")
        else:
            return True

    def load_data(self, X):
        try:
            self.X = X.data
            self.target = X.target

            return self.X
        except:
            raise Exception(
                "X is not a standart sklearn dataset and X.data is not defined")

    def init_centroids(self):
        """
        On initialise les centroids en prenant k points au hasard
        On utilise un random state pour pouvoir reproduire les résultats
        On ajoute un attribut centroids à la classe en utilisant self
        """

        if self.X is None:
            raise Exception(
                "X is not defined - please use .load_data(X) to load data before")

        random_state = np.random.RandomState(self.random_state)
        centroids = self.X[random_state.choice(
            self.X.shape[0], self.k, replace=False)]

        self.centroids = centroids

        return centroids

    def compute_distance(self, X):
        # On construit une matrice de distance entre chaque point et chaque centroid
        # Il faut l'initialiser à 0
        # Les dimensions sont (nombre de points, nombre de centroid)
        # Chaque ligne correspond à un point et chaque colonne à un centroid
        # Pour chaque point, on calcule la distance avec chaque centroid
        # On utilise la norme L2

        distance = np.zeros((X.shape[0], self.k))
        for i, point in enumerate(X):
            distance[i] = norm(point - self.centroids, axis=1)

        self.distance = distance
        
        return self.distance

    def find_cluster_label(self):

        # First we find the minimum distance for each point
        self.cluster = np.argmin(self.distance, axis=1)
        # print("------------------------ FIND CLUSTER LABEL ----------------------")
        # print("CLUSTER : ", self.cluster)

        return self.cluster

    def compute_centroids(self):
        # On sauvegarde les anciens centroids pour pouvoir les comparer
        # On calcule les nouveaux centroids en prenant la moyenne des points de chaque cluster
        # On initialise les nouveaux centroids à 0 par un array de la bonne taille : n_cluster / n_features
        # On effectue une boucle sur les clusters et on calcule la moyenne des points de chaque cluster

        self.old_centroids = self.centroids.copy()
        self.centroids = np.zeros((self.k, self.X.shape[1]))
        for k in range(self.k):
            self.centroids[k] = np.mean(self.X[self.cluster == k], axis=0)

        return self.old_centroids, self.centroids

    def fit(self, X, treshold=0.00001):

        data = self.load_data(X)
        centroids = self.init_centroids()

        for i in range(self.max_iter):
            print("Iteration --> ", i, " <-- / ", self.max_iter, " ...")
            self.compute_distance(data)
            self.find_cluster_label()
            self.compute_centroids()

            self.history[i] = {
                "centroids": self.centroids,
                "cluster": self.cluster,
                "distance": self.distance,
            }

            if abs(np.mean(self.centroids-self.old_centroids)) < treshold:
                break

    # Cette fonction attend un dataframe de la forme IRIS pour avoir le .data et .target

    def plot_results(self, title):

        print("------------------------------------------------------------------------------------------------------")
        print("---------------------------------------------- RESULTS -----------------------------------------------")
        print("------------------------------------------------------------------------------------------------------")

        # To show only 2 dimensions, we use PCA
        # First we fit the PCA on the data, then we transform the data and the centroids
        # Finally we plot the data and the centroids
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        centroids_pca = pca.transform(self.centroids)

        df_X = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_centroids = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])

        df_X['label'] = pd.Series(self.cluster, index=df_X.index)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            ax=ax,
            x='PC1',
            y='PC2',
            hue='label',
            data=df_X,
            palette='tab10',
            marker='o',
            label='data',
        )
        sns.scatterplot(
            ax=ax,
            x='PC1',
            y='PC2',
            hue=df_centroids.index,
            data=df_centroids,
            palette='tab10',
            s=200,
            marker='v',
            alpha=.5,
            label='Centroids',
            legend=False
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f"{title} with k={self.k}")

        plt.show()

    def display_animation(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):

            print("Update called", frame, " / ", len(self.history), " ...")
            centroids = self.history[frame]['centroids']
            labels = self.history[frame]['cluster']
            data = self.X

            pca = PCA(n_components=2)

            data_PCA = pca.fit_transform(data)
            centroids_pca = pca.transform(centroids)

            df_data = pd.DataFrame(data_PCA, columns=['PC1', 'PC2'])
            df_centroids = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])

            df_data['label'] = pd.Series(labels, index=df_data.index)

            ax.clear()

            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                hue='label',
                data=df_data,
                palette='tab10',
                marker='o',
                label='data',
            )

            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                hue=df_centroids.index,
                data=df_centroids,
                palette='tab10',
                s=200,
                marker='v',
                alpha=.5,
                label='Centroids',
                legend=False
            )

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(handles=handles[1:], labels=labels[1:])
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(f"Animation frame {frame} / {len(self.history)}")

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), interval=500)
        plt.show()
        #
