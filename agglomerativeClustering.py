"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import pandas as pd
import datasetProcessing


def main():
    # DatasetProcessing.clean_dataset()

    cc_idx, cc_data = datasetProcessing.extract_cc()
    covid_idx, covid_data = datasetProcessing.extract_covid()
    cccd_idx, cccd_data = datasetProcessing.extract_cccd()
    mall_idx, mall_data = datasetProcessing.extract_mall()
    stc_idx, stc_data = datasetProcessing.extract_stc()
    wcd_idx, wcd_data = datasetProcessing.extract_wcd()

    result_df = pd.DataFrame(columns = ['Name', 'Num of Dimensions', 'Linkage', 'Distance', 'Num of clusters',
                                        'Silhouette Score', 'Calinski Harabasz Score', 'Davies-Bouldin Index',
                                        'Runtime'])
    model = result_df.copy()

    i = 0
    cc_df = model.copy()
    result_df, i, cc_df = datasetProcessing.show_result(cc_data, 'CC General', result_df, i, cc_df, 0)
    cc_df.to_csv('Datasets/Dataframes/CC_df.csv', index = False)
    cc_df.to_csv('Datasets/Dataframes/CC_df_idx.csv', index = True)
    print('\n')
    covid_df = model.copy()
    result_df, i, covid_df = datasetProcessing.show_result(covid_data, 'COVID-19', result_df, i, covid_df, 0)
    covid_df.to_csv('Datasets/Dataframes/COVID_df.csv', index = False)
    covid_df.to_csv('Datasets/Dataframes/COVID_df_idx.csv', index = True)
    print('\n')
    cccd_df = model.copy()
    result_df, i, cccd_df = datasetProcessing.show_result(cccd_data, 'Credit Card', result_df, i, cccd_df, 0)
    cccd_df.to_csv('Datasets/Dataframes/Credit_Card_df.csv', index = False)
    cccd_df.to_csv('Datasets/Dataframes/Credit_Card_df_idx.csv', index = True)
    print('\n')
    mall_df = model.copy()
    result_df, i, mall_df = datasetProcessing.show_result(mall_data, 'Mall Customer', result_df, i, mall_df, 0)
    mall_df.to_csv('Datasets/Dataframes/Mall_df.csv', index = False)
    mall_df.to_csv('Datasets/Dataframes/Mall_df_idx.csv', index = True)
    print('\n')
    stc_df = model.copy()
    result_df, i, stc_df = datasetProcessing.show_result(stc_data, 'Sales Transactions', result_df, i, stc_df, 0)
    stc_df.to_csv('Datasets/Dataframes/Sales_Transactions_df.csv', index = False)
    stc_df.to_csv('Datasets/Dataframes/Sales_Transactions_df_idx.csv', index = True)
    print('\n')
    wcd_df = model.copy()
    result_df, i, wcd_df = datasetProcessing.show_result(wcd_data, 'Wholesale', result_df, i, wcd_df, 0)
    wcd_df.to_csv('Datasets/Dataframes/Wholesale_df.csv', index = False)
    wcd_df.to_csv('Datasets/Dataframes/Wholesale_df_idx.csv', index = True)

    result_df.to_csv('Datasets/Dataframes/Results.csv', index = False)
    result_df.to_csv('Datasets/Dataframes/Results_idx.csv', index = True)


if __name__ == '__main__':
    main()
