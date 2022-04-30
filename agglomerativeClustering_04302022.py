"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import DatasetProcessing


def main():
    #DatasetProcessing.clean_dataset()

    cc_idx, cc_data = DatasetProcessing.extract_cc()
    covid_idx, covid_data = DatasetProcessing.extract_covid()
    cccd_idx, cccd_data = DatasetProcessing.extract_cccd()
    mall_idx, mall_data = DatasetProcessing.extract_mall()
    stc_idx, stc_data = DatasetProcessing.extract_stc()
    wcd_idx, wcd_data = DatasetProcessing.extract_wcd()

    DatasetProcessing.show_result(cc_data, 'CC General')
    print('\n')
    DatasetProcessing.show_result(covid_data, 'COVID-19')
    print('\n')
    DatasetProcessing.show_result(cccd_data, 'Credit Card')
    print('\n')
    DatasetProcessing.show_result(mall_data, 'Mall Customer')
    print('\n')
    DatasetProcessing.show_result(stc_data, 'Sales Transactions')
    print('\n')
    DatasetProcessing.show_result(wcd_data, 'Wholesale')


if __name__ == '__main__':
    main()