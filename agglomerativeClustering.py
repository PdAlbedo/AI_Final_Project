"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import datasetProcessing


def main():
    # DatasetProcessing.clean_dataset()

    cc_idx, cc_data = datasetProcessing.extract_cc()
    covid_idx, covid_data = datasetProcessing.extract_covid()
    cccd_idx, cccd_data = datasetProcessing.extract_cccd()
    mall_idx, mall_data = datasetProcessing.extract_mall()
    stc_idx, stc_data = datasetProcessing.extract_stc()
    wcd_idx, wcd_data = datasetProcessing.extract_wcd()

    datasetProcessing.show_result(cc_data, 'CC General')
    print('\n')
    datasetProcessing.show_result(covid_data, 'COVID-19')
    print('\n')
    datasetProcessing.show_result(cccd_data, 'Credit Card')
    print('\n')
    datasetProcessing.show_result(mall_data, 'Mall Customer')
    print('\n')
    datasetProcessing.show_result(stc_data, 'Sales Transactions')
    print('\n')
    datasetProcessing.show_result(wcd_data, 'Wholesale')


if __name__ == '__main__':
    main()
