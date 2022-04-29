"""
Pre-compute the dataset by cleaning or other operations
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_dataset():
    df_cc = pd.read_csv('Datasets/CC_GENERAL.csv', header = 0)
    del df_cc['CREDIT_LIMIT']
    del df_cc['MINIMUM_PAYMENTS']
    df_cc.to_csv('Datasets/CleanedDatasets/Cleaned_CC_GENERAL.csv', index = False)

    df_covid = pd.read_csv('Datasets/COVID-19-Coronavirus.csv', header = 0)
    del df_covid['Other names']
    del df_covid['ISO 3166-1 alpha-3 CODE']
    del df_covid['Continent']
    df_covid.to_csv('Datasets/CleanedDatasets/Cleaned_COVID.csv', index = False)

    df_cccd = pd.read_csv('Datasets/Credit_Card_Customer_Data.csv', header = 0)
    del df_cccd['Customer Key']
    df_cccd.to_csv('Datasets/CleanedDatasets/Cleaned_Credit_Card_Customer_Data.csv', index = False)

    df_mall = pd.read_csv('Datasets/Mall_Customers.csv', header = 0)
    df_mall['Gender'] = df_mall['Gender'].map(lambda x: 0 if x == 'Male' else x)
    df_mall['Gender'] = df_mall['Gender'].map(lambda x: 1 if x == 'Female' else x)
    df_mall.to_csv('Datasets/CleanedDatasets/Cleaned_Mall_Customers.csv', index = False)

    df_std = pd.read_csv('Datasets/Sales_Transactions_Dataset_Weekly.csv', header = 0)
    df_std = df_std.drop(df_std.columns[[i for i in range(1, 55)]], axis = 1)
    df_std.to_csv('Datasets/CleanedDatasets/Cleaned_Sales_Transactions.csv', index = False)

    df_wcd = pd.read_csv('Datasets/Wholesale_customers_data.csv', header = 0)
    df_wcd.to_csv('Datasets/CleanedDatasets/Cleaned_Wholesale.csv', index = False)


def extract_cc():
    df_cc_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_CC_GENERAL.csv', header = 0)
    cc_idx = df_cc_raw['CUST_ID']
    cc_data = df_cc_raw[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                         'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                         'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX',
                         'PURCHASES_TRX', 'PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]
    cc_data = StandardScaler().fit_transform(cc_data)
    cc_data = MinMaxScaler().fit_transform(cc_data)
    df_cc_norm = pd.DataFrame(data = cc_data, columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
                                                         'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                                                         'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                                                         'ONEOFF_PURCHASES_FREQUENCY',
                                                         'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
                                                         'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
                                                         'PRC_FULL_PAYMENT', 'TENURE'])
    return cc_idx, df_cc_norm


def extract_covid():
    df_covid_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_COVID.csv', header = 0)
    covid_idx = df_covid_raw['Country']
    covid_data = df_covid_raw[df_covid_raw.columns[[i for i in range(1, 7)]]]
    covid_data = StandardScaler().fit_transform(covid_data)
    covid_data = MinMaxScaler().fit_transform(covid_data)
    df_covid_norm = pd.DataFrame(data = covid_data, columns = ['Population',
                                                               'Total Cases', 'Total Deaths', 'Tot Cases//1M pop',
                                                               'Tot Deaths/1M pop', 'Death percentage'])
    return covid_idx, df_covid_norm


def extract_cccd():
    df_cccd_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_Credit_Card_Customer_Data.csv', header = 0)
    cccd_idx = df_cccd_raw['Sl_No']
    cccd_data = df_cccd_raw[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online',
                             'Total_calls_made']]
    cccd_data = StandardScaler().fit_transform(cccd_data)
    cccd_data = MinMaxScaler().fit_transform(cccd_data)
    df_cccd_norm = pd.DataFrame(data = cccd_data, columns = ['Avg_Credit_Limit', 'Total_Credit_Cards',
                                                             'Total_visits_bank', 'Total_visits_online',
                                                             'Total_calls_made'])
    return cccd_idx, df_cccd_norm


def extract_mall():
    df_mall_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_Mall_Customers.csv', header = 0)
    mall_idx = df_mall_raw['CustomerID']
    mall_data = df_mall_raw[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    mall_data = StandardScaler().fit_transform(mall_data)
    mall_data = MinMaxScaler().fit_transform(mall_data)
    df_mall_norm = pd.DataFrame(data = mall_data, columns = ['Gender', 'Age', 'Annual Income (k$)',
                                                             'Spending Score (1-100)'])
    return mall_idx, df_mall_norm


def extract_stc():
    df_stc_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_Sales_Transactions.csv', header = 0)
    stc_idx = df_stc_raw['Product_Code']
    stc_data = df_stc_raw[df_stc_raw.columns[[i for i in range(1, 53)]]]
    stc_data = StandardScaler().fit_transform(stc_data)
    stc_data = MinMaxScaler().fit_transform(stc_data)
    df_stc_norm = pd.DataFrame(data = stc_data, columns = ['Normalized 0', 'Normalized 1', 'Normalized 2',
                                                           'Normalized 3', 'Normalized 4', 'Normalized 5',
                                                           'Normalized 6', 'Normalized 7', 'Normalized 8',
                                                           'Normalized 9', 'Normalized 10', 'Normalized 11',
                                                           'Normalized 12', 'Normalized 13', 'Normalized 14',
                                                           'Normalized 15', 'Normalized 16', 'Normalized 17',
                                                           'Normalized 18', 'Normalized 19', 'Normalized 20',
                                                           'Normalized 21', 'Normalized 22', 'Normalized 23',
                                                           'Normalized 24', 'Normalized 25', 'Normalized 26',
                                                           'Normalized 27', 'Normalized 28', 'Normalized 29',
                                                           'Normalized 30', 'Normalized 31', 'Normalized 32',
                                                           'Normalized 33', 'Normalized 34', 'Normalized 35',
                                                           'Normalized 36', 'Normalized 37', 'Normalized 38',
                                                           'Normalized 39', 'Normalized 40', 'Normalized 41',
                                                           'Normalized 42', 'Normalized 43', 'Normalized 44',
                                                           'Normalized 45', 'Normalized 46', 'Normalized 47',
                                                           'Normalized 48', 'Normalized 49', 'Normalized 50',
                                                           'Normalized 51'])
    return stc_idx, df_stc_norm


def extract_wcd():
    df_wcd_raw = pd.read_csv('Datasets/CleanedDatasets/Cleaned_Wholesale.csv', header = 0)
    wcd_idx = df_wcd_raw[['Channel', 'Region']]
    wcd_data = df_wcd_raw[['Fresh', 'Milk', 'Grocery', 'Frozen',
                           'Detergents_Paper', 'Delicassen']]
    wcd_data = StandardScaler().fit_transform(wcd_data)
    wcd_data = MinMaxScaler().fit_transform(wcd_data)
    df_wcd_norm = pd.DataFrame(data = wcd_data, columns = ['Fresh', 'Milk', 'Grocery', 'Frozen',
                                                           'Detergents_Paper', 'Delicassen'])
    return wcd_idx, df_wcd_norm


def show_result(df, name):
    # TODO
    # pca = None
    # df_pca = None
    is_passed = False
    case = 0
    # TODO
    # for i in range(2, 5):
    # pca = PCA(n_components = i).fit_transform(df)
    for linkage in ("ward", "average", "complete", "single"):
        for dis_matrix in ("euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"):
            if linkage != "ward":
                is_passed = False
            if is_passed:
                continue
            if linkage == "ward":
                dis_matrix = "euclidean"
                is_passed = True
            # TODO
            # if i == 2:
            #     df_pca = pd.DataFrame(data = pca_df, columns = ['Pc 1', 'PC 2'])
            # elif i == 3:
            #     df_pca = pd.DataFrame(data = pca_df, columns = ['PC 1', 'PC 2', 'PC 3'])
            # elif i == 4:
            #     df_pca = pd.DataFrame(data = pca_df, columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4'])
            t0 = time()
            if dis_matrix != "precomputed":
                clustering = AgglomerativeClustering(affinity = dis_matrix, linkage = linkage,
                                                     distance_threshold = 0.01, n_clusters = None)
                clustering.fit(df)
                print('###########################################################')
                print('%s case %d: ' % (name, case))
                print('Num of clusters: %d' % max(clustering.labels_))
                print(clustering.labels_)
                print("Dimension after PCA:%d\tLinkage: %s\tDistance: %s\tRuntime: %.2fs" %
                      (0, linkage, dis_matrix, time() - t0))
                case += 1
                print('###########################################################\n')
            else:
                for i in range(2):
                    x = df
                    if i == 0:
                        dist = DistanceMetric.get_metric('chebyshev')
                        dis_name = 'chebyshev'
                    if i == 1:
                        dist = DistanceMetric.get_metric('minkowski')
                        dis_name = 'minkowski'
                    # TODO
                    # if i == 2:
                    #     dist = DistanceMetric.get_metric('wminkowski')
                    #     dis_name = 'wminkowski'
                    # else:
                    #     dist = DistanceMetric.get_metric('mahalanobis', V = np.cov(x))
                    #     dis_name = 'mahalanobis'

                    precomputedMatrix = dist.pairwise(x)

                    clustering = AgglomerativeClustering(affinity = dis_matrix, linkage = linkage,
                                                         distance_threshold = 0.01, n_clusters = None)
                    clustering.fit(precomputedMatrix)
                    print('###########################################################')
                    print('%s case %d: ' % (name, case))
                    print('Num of clusters: %d' % max(clustering.labels_))
                    print(clustering.labels_)
                    print("Dimension after PCA:%d\tLinkage: %s\tDistance: %s\tRuntime: %.2fs" %
                          (0, linkage, dis_name, time() - t0))
                    case += 1
                    print('###########################################################\n')