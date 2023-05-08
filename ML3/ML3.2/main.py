# We are going to implement a kmeans algorithm
# We will code each function separately and then put them together
# we will build a class to hold the functions
from sklearn.datasets import load_iris, load_wine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from KmeansClassifier import KmeansClassifier
from sklearn.decomposition import PCA
import matplotlib.animation as animation

iris = load_iris()
wine = load_wine()

clf = KmeansClassifier(random_state=42, k=3, max_iter=100)
clf.fit(iris)

clf.display_animation()