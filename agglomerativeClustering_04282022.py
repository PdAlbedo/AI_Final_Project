"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import string
from time import time
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.neighbors import DistanceMetric


digits = datasets.load_digits()
X, y = digits.data, digits.target
n_samples, n_features = X.shape

np.random.seed(0)
# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title = None):
    x_min, x_max = np.min(X_red, axis = 0), np.max(X_red, axis = 0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize = (6, 4))
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,
            marker = f"${digit}$",
            s = 50,
            c = plt.cm.nipy_spectral(labels[y == digit] / 10),
            alpha = 0.5,
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size = 17)
    plt.axis("off")
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])

def plot_some():
    # ----------------------------------------------------------------------
    # 2D embedding of the digits dataset
    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components = 2).fit_transform(X)
    print("Done.")

    for linkage in ("ward", "average", "complete", "single"):
        clustering = AgglomerativeClustering(linkage = linkage, n_clusters = None, affinity='euclidean')
        t0 = time()
        clustering.fit(X_red)
        print("%s :\t%.2fs" % (linkage, time() - t0))

        plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)

    plt.show()

    return



def main():
    input_file = 'Mall_Customers.csv'
    df = pd.read_csv(input_file, header = 0)

    # convert gender to numeric data
    df['Gender'] = df['Gender'].map(lambda x: 0 if x == 'Male' else x)
    df['Gender'] = df['Gender'].map(lambda x: 1 if x == 'Female' else x)

    # display some basic info of dataset
    original_header = list(df.columns.values)
    print("Categories: ", end = "")
    print(original_header)
    print("\nTop 10 entries: ")
    print(df.head(10))
    print("\nShape of dateset: ", end = "")
    print(df.shape)

    df_age_annual_income = df[['Age', 'Annual Income (k$)']].copy()
    print(df_age_annual_income.head(10))
    
    # print("Computing embedding")
    # embedded = manifold.SpectralEmbedding(n_components = 2).fit_transform(df_age_annual_income)
    # print("Done.")


    X = df
    #dist = DistanceMetric.get_metric('minkowski')
    dist = DistanceMetric.get_metric('mahalanobis', V = np.cov(X, rowvar=False))
    

    #precomputedMatrix = dist.pairwise(X)
    precomputedMatrix = dist.pairwise(X)

    print("Precomputed Matrix: ", precomputedMatrix)

    for linkage in ("average", "complete", "single"):
        #clustering = AgglomerativeClustering(linkage = linkage, n_clusters = 10, affinity='euclidean')
        clustering = AgglomerativeClustering(linkage = linkage, n_clusters = 10, affinity='precomputed')
        t0 = time()
        clustering.fit(precomputedMatrix)
        print(clustering.labels_)
        print("%s :\t%.2fs" % (linkage, time() - t0))

        # plt.figure(figsize = (6, 4))
        # for label in clustering.labels_:
        #     plt.scatter(
        #         df_age_annual_income['Age'],
        #         df_age_annual_income['Annual Income (k$)'],
        #         marker = f"${label}$",
        #         s = 50,
        #         c = plt.cm.nipy_spectral(label * 10),
        #         alpha = 0.5,
        #     )

        plt.xticks([])
        plt.yticks([])

        plt.show()
    # 
    #     plot_clustering(embedded, clustering.labels_, "%s linkage" % linkage)
    # 
    # plt.show()
    # numpy_array = df.as_matrix()
    # numeric_headers.reverse()
    # reverse_df = df[numeric_headers]
    # reverse_df.to_excel('path_to_file.xls')





if __name__ == '__main__':
    main()
