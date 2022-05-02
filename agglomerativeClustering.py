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

    i = 0
    result_df, i = datasetProcessing.show_result(cc_data, 'CC General', result_df, i)
    print('\n')
    result_df, i = datasetProcessing.show_result(covid_data, 'COVID-19', result_df, i)
    print('\n')
    result_df, i = datasetProcessing.show_result(cccd_data, 'Credit Card', result_df, i)
    print('\n')
    result_df, i = datasetProcessing.show_result(mall_data, 'Mall Customer', result_df, i)
    print('\n')
    result_df, i = datasetProcessing.show_result(stc_data, 'Sales Transactions', result_df, i)
    print('\n')
    result_df, i = datasetProcessing.show_result(wcd_data, 'Wholesale', result_df, i)

    result_df.to_csv('Results.csv', index = False)


if __name__ == '__main__':
    main()
