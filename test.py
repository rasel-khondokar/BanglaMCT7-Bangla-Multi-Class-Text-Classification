from scraping.banglatribune import main_banglatribune
from scraping.jagonews24 import main_jagonews24
from scraping.multi_threding import scrape_jagonews24
from scraping.prothom_alo import main_prothom_alo

categories = {'sports':'sports', 'international':'international', 'economy':'economy', 'entertainment':'entertainment',
                         'technology':'technology', 'politics':'politics', 'education':'education'}

for category in categories:
    main_jagonews24(categories, category)

# from itertools import chain
#
# import numpy as np
# import pandas as pd
#
# from configs import BASE_DIR
# from exploratory_analysis.eda import EDA
# from preprocessing.preprocessing import PreProcessor
# from settings import DIR_PERFORMENCE_REPORT
#
#
# def make_df_from_cls_report(filename):
#     with open(f'{DIR_PERFORMENCE_REPORT}{filename}.txt') as file:
#         text = file.read()
#     text = text.split('\n')[12:20]
#     data = []
#     for i in text:
#         d = [j for j in i.split(' ') if j]
#         data.append(d)
#     data = pd.DataFrame(data, columns=['class', 'precision', 'recall', 'f1-score', 'support'])
#     data.iloc[0, 0] = filename[30:]
#     return data
#
# def make_excel_report_from_cls_report(name, files):
#     df = pd.DataFrame()
#     for f in files:
#         df = df.append(make_df_from_cls_report(f), ignore_index=True)
#     df.to_excel(f'{DIR_PERFORMENCE_REPORT}performence table - {name}.xlsx')
# # make_excel_report_from_cls_report('prothomalo', [
# #     'test_others_prothomalo_rm_oth_automl_best_model_LinearSVC',
# #     'test_others__prothomalo_rm_oth_incorrect_fasttext_bi_lstm',
# #     'test_others__prothomalo_rm_oth_incorrect_fasttext_simple',
# #     'test_others__prothomalo_rm_oth_incorrect_keras_cnn_bi_lstm',
# #     'test_others__prothomalo_rm_oth_incorrect_keras_bi_gru',
# #     'test_others_prothomalo_rm_oth_bert-base-multilingual-cased_train',
# #     'test_others_prothomalo_rm_oth_csebuetnlp_banglabert_train',
# #     'test_others_prothomalo_rm_oth_monsoon-nlp_bangla-electra_train',
# #     'test_others_prothomalo_rm_oth_sagorsarker_bangla-bert-base_train'
# # ])
#
# # preprocessor = PreProcessor()
# # data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()
# # data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()
# # # # exploratory data analysis
# # eda = EDA(data, 'test')
# # eda.analyze()
# def get_len(x):
#     return len(x)
#
# # data['l_r'] = data['cleanText'].apply(get_len)
# # data['l_c'] =  data['cleaned'].apply(get_len)
# # data['dif'] = data['l_r']-data['l_c']
# # data = data.sort_values(by=['dif'])
# #
# # for i in range(len(data)):
# #     print('___________________________________________________________________________________________________________________________________________________________')
# #     print(data.iloc[i]['dif'])
# #     print('raw : ')
# #     print(data.iloc[i]['cleanText'])
# #     print('clean : ')
# #     print(data.iloc[i]['cleaned'])
# #     print(data.iloc[i]['category'])
#
#
#
# dataset_dir = f'{BASE_DIR}/DATASET/'
# file = f'{dataset_dir}collected_removed_urls_incorrect.csv'
# dataset = pd.read_csv(file)
# # dataset = dataset.sample(100)
# dataset.dropna(inplace=True)
# dataset = dataset.drop_duplicates(subset=['url'])
# cleanText = dataset['cleanText'].to_list()
# word_list = [word.strip().lower() for t in cleanText for word in
#                          t.strip().split()]
# # flatten_list = list(chain.from_iterable(word_list))
# print("Number of Unique Words:{}".format(len(set(word_list))))
# print("Number of Unique Words:{}".format(len(np.unique(word_list))))
#
