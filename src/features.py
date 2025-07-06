import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans



# Removing Outliers
def remove_outliers_iqr(df, columns=None, factor=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean



# Encoding selected Features
def encode_features(df, categorical_cols, len=1):
    le = LabelEncoder()

    if len>1:
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
        return df
    else:
        categorical_cols = le.fit_transform(categorical_cols)
        return categorical_cols



# Scaling data
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)



# Elbow Optimization
def elbow(X, range_end):
    inertia = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, 
                        init='k-means++',   # k-means++ speeds up convergence by spreading out the initial centroids as far as possible from one another.
                        max_iter=300, 
                        n_init=10, 
                        random_state=0
        )
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plotting Elbow graph
    plt.figure(figsize=(12,6))
    plt.plot(range(1,range_end), 
             inertia, 
             marker='o',
             linestyle='--', 
             color='navy'
    )
    plt.title('Elbow Method: Optimization')
    plt.xlabel('No of Cluster')
    plt.ylabel('Inertia')

    plt.show()
