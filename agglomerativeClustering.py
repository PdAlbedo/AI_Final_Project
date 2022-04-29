"""
Apply agglomerative clustering to the given dataset and find out optimal
distance measure for agglomerative hierarchical clustering
"""
__author__ = "Xichen Liu, Jeff Turgeon"

import DatesetProcessing


def main():
    # DatesetProcessing.clean_dataset()

    cc_idx, cc_data = DatesetProcessing.extract_cc()
    covid_idx, covid_data = DatesetProcessing.extract_covid()
    cccd_idx, cccd_data = DatesetProcessing.extract_cccd()
    mall_idx, mall_data = DatesetProcessing.extract_mall()
    stc_idx, stc_data = DatesetProcessing.extract_stc()
    wcd_idx, wcd_data = DatesetProcessing.extract_wcd()

    DatesetProcessing.show_result(cc_data, 'CC General')
    print('\n')
    DatesetProcessing.show_result(covid_data, 'COVID-19')
    print('\n')
    DatesetProcessing.show_result(cccd_data, 'Credit Card')
    print('\n')
    DatesetProcessing.show_result(mall_data, 'Mall Customer')
    print('\n')
    DatesetProcessing.show_result(stc_data, 'Sales Transactions')
    print('\n')
    DatesetProcessing.show_result(wcd_data, 'Wholesale')


if __name__ == '__main__':
    main()
