"""
Pre-compute the dataset by cleaning or other operations
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


def pre_compute_dataset():
    df_mall = pd.read_csv('Datasets/Mall_Customers.csv', header = 0)
    df_mall['Gender'] = df_mall['Gender'].map(lambda x: 0 if x == 'Male' else x)
    df_mall['Gender'] = df_mall['Gender'].map(lambda x: 1 if x == 'Female' else x)
    df_mall.to_csv('Datasets/Cleaned_Mall_Customers.csv', index = False)

    # df_cc = pd.read_csv('Datasets/CC_GENERAL.csv', header = 0)
    # del df_cc['CREDIT_LIMIT']
    # del df_cc['MINIMUM_PAYMENTS']
    # df_cc.to_csv('Datasets/Cleaned_CC_GENERAL.csv', index = False)

    df_cc = pd.read_csv('Datasets/CC_GENERAL.csv', header = 0)
    df_cc['CUST_ID'] = df_cc['CUST_ID'].apply(lambda x: x[1:])
    del df_cc['CREDIT_LIMIT']
    del df_cc['MINIMUM_PAYMENTS']
    df_cc.to_csv('Datasets/Cleaned_CC_GENERAL.csv', index = False)

    df_cccd = pd.read_csv('Datasets/Credit_Card_Customer_Data.csv', header = 0)
    del df_cccd['Customer Key']
    df_cccd.to_csv('Datasets/Cleaned_Credit_Card_Customer_Data.csv', index = False)


def show_result(df, name):
    # pca_df = None TODO
    x = None
    is_passed = False
    case = 0
    for i in range(2, 5):
        # pca_df = PCA(n_components = i).fit_transform(df)
        for linkage in ("ward", "average", "complete", "single"):
            for dis_matrix in ("euclidean", "l1", "l2", "manhattan", "cosine"):
                if linkage != "ward":
                    is_passed = False
                if is_passed:
                    continue
                if linkage == "ward":
                    dis_matrix = "euclidean"
                    is_passed = True
                # if i == 2:
                #     x = pd.DataFrame(data = pca_df, columns = ['Pc 1', 'PC 2'])
                # elif i == 3:
                #     x = pd.DataFrame(data = pca_df, columns = ['PC 1', 'PC 2', 'PC 3'])
                # elif i == 4:
                #     x = pd.DataFrame(data = pca_df, columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4'])
                t0 = time()
                clustering = AgglomerativeClustering(affinity = dis_matrix, linkage = linkage,
                                                     distance_threshold = 0.1, n_clusters = None)
                clustering.fit(df)
                print('###########################################################')
                print('%s case %d: ' % (name, case))
                print(clustering.labels_)
                print("Dimension after PCA:%d Linkage: %s Distance: %s\t%.2fs" %
                      (i, linkage, dis_matrix, time() - t0))
                case += 1
                print('###########################################################\n')
