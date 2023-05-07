import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from numpy.linalg import norm



class KmeansClassifier:
    def __init__(self, k=3, max_iter=100, X=None, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

    def __str__(self) -> str:
        return f"KmeansClassifier(k={self.k}, max_iter={self.max_iter})"

    def check_data(self):
        if not self.X:
            raise Exception(
                "X is not defined - please use .load_data(X) to load data before")
        else:
            return True

    def load_data(self, X):
        if X is not None:
            self.X = X.data
            self.target = X.target
            
            return self.X
        else:
            raise Exception("X cannot be None")

    def init_centroids(self):
        # On initialise les centroids en prenant k points au hasard
        # On utilise un random state pour pouvoir reproduire les résultats
        # On ajoute un attribut centroids à la classe en utilisant self
        
        if self.X is None:
            raise Exception("X is not defined - please use .load_data(X) to load data before")
        
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
        distance = np.zeros((X.shape[0], self.k))
        #print("------------------------ COMPUTATION OF DISTANCE ----------------------")
        #print("-----------------------------------------------------------------------")
        
        # Pour chaque point, on calcule la distance avec chaque centroid
        # On utilise la norme L2
        
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
        self.old_centroids = self.centroids.copy()
        
        # On calcule les nouveaux centroids en prenant la moyenne des points de chaque cluster
        # On initialise les nouveaux centroids à 0 par un array de la bonne taille : n_cluster / n_features
        # On effectue une boucle sur les clusters et on calcule la moyenne des points de chaque cluster
        
        
        self.centroids = np.zeros((self.k, self.X.shape[1]))
        for k in range(self.k):
            
            # print(f"Cluster {k} --> {self.X[self.cluster == k]}")
            self.centroids[k] = np.mean(self.X[self.cluster == k], axis=0)
            
            
        #print("------------------------ COMPUTE CENTROIDS ----------------------")
        #print("CENTROIDS : ", self.centroids)
        #print("OLD CENTROIDS : ", self.old_centroids)
        
        return self.old_centroids, self.centroids
    
    def fit(self, X, treshold=0.00001):
        
        data = self.load_data(X)
        centroids = self.init_centroids()
        
        for i in range(self.max_iter):
            print("ITERATION : ", i)
            self.compute_distance(data)
            self.find_cluster_label()
            self.compute_centroids()
            
            D_tresh = abs(np.mean(self.centroids-self.old_centroids))
            print("Centroids",self.centroids)
            print("Old Centroids",self.old_centroids)
            print("distance", D_tresh)
            
            
            if D_tresh < treshold:

                print("TRESHOLD REACHED", D_tresh)
                break

    
    
    # Cette fonction attend un dataframe de la forme IRIS pour avoir le .data et .target
    def plot(self):
        
        print("------------------------ PLOT ----------------------")
        

        if self.X is not None:
            if self.centroids is None:
                self.init_centroids()

            pca = PCA(n_components=2)

            X_pca = pca.fit_transform(self.X)
            centroids_pca = pca.transform(self.centroids)
            centroids_pca_old = pca.transform(self.old_centroids)
            
            
            df_X = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_centroids = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])
            
            centroids_pca_old = pca.transform(self.old_centroids)
            df_centroids_old = pd.DataFrame(centroids_pca_old, columns=['PC1', 'PC2'])



            # df_pca['label'] = self.target

            df_X['label'] = pd.Series(self.cluster, index=df_X.index)
            print("------------------------ DF_X ----------------------")
            print(df_X)
            print("------------------------ DF_centroids ----------------------")
            print(df_centroids)
            print("------------------------ DF_centroids_old ----------------------")
            print(df_centroids_old)
            
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
                marker='x',
                alpha=.5,
                label='new_centroids',
            )
            
            sns.scatterplot(
                ax=ax,
                x='PC1',
                y='PC2',
                hue=df_centroids_old.index,
                data=df_centroids_old,
                palette='tab10',
                s=200,
                marker='v',
                alpha=.5,
                label='old_centroids',
            )

            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('PCA on Iris Dataset')
            plt.show()


            
        else:
            raise Exception("X cannot be None")
            