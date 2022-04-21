"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import preComputation
from sklearn.decomposition import PCA
from time import time


def main():
    preComputation.pre_compute_dataset()

    df_mall_raw = pd.read_csv('Datasets/Cleaned_Mall_Customers.csv', header = 0)
    mall_idx = df_mall_raw['CustomerID']
    mall_data = df_mall_raw[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    mall_data = StandardScaler().fit_transform(mall_data)
    df_mall_norm = pd.DataFrame(data = mall_data, columns = ['Gender', 'Age', 'Annual Income (k$)',
                                                             'Spending Score (1-100)'])

    df_cc_raw = pd.read_csv('Datasets/Cleaned_CC_GENERAL.csv', header = 0)
    cc_idx = df_cc_raw['CUST_ID']
    cc_data = df_cc_raw[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                         'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                         'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX',
                         'PURCHASES_TRX', 'PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]
    cc_data = StandardScaler().fit_transform(cc_data)
    df_cc_norm = pd.DataFrame(data = cc_data, columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
                                                         'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                                                         'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                                                         'ONEOFF_PURCHASES_FREQUENCY',
                                                         'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
                                                         'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
                                                         'PRC_FULL_PAYMENT', 'TENURE'])

    df_cccd_raw = pd.read_csv('Datasets/Cleaned_Credit_Card_Customer_Data.csv', header = 0)
    cccd_idx = df_cccd_raw['Sl_No']
    cccd_data = df_cccd_raw[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank',
                             'Total_visits_online', 'Total_calls_made']]
    cccd_data = StandardScaler().fit_transform(cccd_data)
    df_cccd_norm = pd.DataFrame(data = cccd_data, columns = ['Avg_Credit_Limit', 'Total_Credit_Cards',
                                                             'Total_visits_bank', 'Total_visits_online',
                                                             'Total_calls_made'])

    preComputation.show_result(df_mall_norm, 'Mall Customer')
    print('\n')
    preComputation.show_result(df_cc_norm, 'CC General')
    print('\n')
    preComputation.show_result(df_cccd_raw, 'Credit Card')

    # pca_df = PCA(n_components = 2).fit_transform(df_cc_norm)
    # clustering = AgglomerativeClustering(affinity = "euclidean", linkage = "complete",
    #                                      distance_threshold = 15, n_clusters = None)
    # clustering.fit(pca_df)
    # print(clustering.labels_.shape)


if __name__ == '__main__':
    main()
