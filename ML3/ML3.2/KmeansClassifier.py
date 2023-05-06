import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from numpy.linalg import norm



class KmeansClassifier:
    def __init__(self, k=3, max_iter=100, X=None):
        self.k = k
        self.max_iter = max_iter

    def __str__(self) -> str:
        return f"KmeansClassifier(k={self.k}, max_iter={self.max_iter})"
    
    


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
    
    def compute_distance(self):
        
        # On construit une matrice de distance entre chaque point et chaque centroid
        # Il faut l'initialiser à 0
        # Les dimensions sont (nombre de points, nombre de centroid)
        # Chaque ligne correspond à un point et chaque colonne à un centroid
        distance = np.zeros((self.X.shape[0], self.k))
        print("------------------------ COMPUTATION OF DISTANCE ----------------------")
        print("-----------------------------------------------------------------------")
        # print(distance[2])
        
        for i, point in enumerate(self.X):
            # distance[point,:] = norm(point - self.centroids, axis=1)
            # print("POINT : ", point)
            for cluster in self.centroids:
                print("CLUSTER : ", cluster)
            # print(norm(point-self.centroids, axis=1))
            distance[i] = norm(point - self.centroids, axis=1)
            # print(distance)
            
        self.distance = distance
            
        return distance
    
    def find_cluster_label(self):

        # First we find the minimum distance for each point
        self.cluster = np.argmin(self.distance, axis=1)
        
        return self.cluster
    
    
    # Cette fonction attend un dataframe de la forme IRIS pour avoir le .data et .target

    def plot(self):
        
        print("--------------------- PLOT ---------------------")
        

        if self.X is not None:
            if self.centroids is not None:
                self.init_centroids()

            pca = PCA(n_components=2)

            X_pca = pca.fit_transform(self.X)
            centroids_pca = pca.transform(self.centroids)

            df_X = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_centroids = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])

            # df_pca['label'] = self.target

            df_X['label'] = pd.Series(self.cluster, index=df_X.index)
            print(df_X)
            print(df_centroids)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                hue='label',
                data=df_X,
                palette='tab10',
                marker='o',
            )

            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                hue=df_centroids.index,
                data=df_centroids,
                palette='tab10',
                s=200,
                marker='x',
                alpha=.5,
            )

            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('PCA on Iris Dataset')
            plt.show()


            
        else:
            raise Exception("X cannot be None")
            