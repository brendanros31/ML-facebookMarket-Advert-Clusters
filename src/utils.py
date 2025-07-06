import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


# PCA Cluster plots
def plot_clusters(trained_model, X, n_components=2):
    labels =  trained_model.labels_
    
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    # Plotting PCA
    plt.figure(figsize=(12,6))
    scatter = plt.scatter(X_pca[:,0], 
                          X_pca[:,1], 
                          c=labels, 
                          cmap='tab10', 
                          alpha=0.7
    )
    plt.title('Cluster Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.colorbar(scatter, label='Cluster')
    plt.show()


