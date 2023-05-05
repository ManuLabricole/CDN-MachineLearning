import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class KmeansClassifier:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def __str__(self) -> str:
        return f"KmeansClassifier(k={self.k}, max_iter={self.max_iter})"
    
    

    def check_input(self, X, y):
        if X is not None and y is not None:
            self.X = X
            self.y = y
            print(f"X : {X}, y : {y}")
        else:
            raise Exception("X and y cannot be None")

    def plot(self):
        
        fig, ax = plt.subplots(figsize=(12, 8))
        # Create a dataframe with the principal components
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

        # Add a target column to the dataframe
        df_pca['target'] = iris.target

        # Replace target values with class names
        df_pca['target'] = df_pca['target'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

        # Create a scatter plot of the first two principal components
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca, palette='Set1')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA on Iris Dataset')
        plt.show()
        
    
