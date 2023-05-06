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

clf.load_data(iris)
clf.init_centroids()
clf.compute_distance()
cluster = clf.find_cluster_label()
clf.plot()
