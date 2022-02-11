import json

import pandas as pd

from preprocessing.preprocessing import PreProcessor


def get_incorrect_news_urls(file):
    our_fasttext_simple = pd.read_csv(f'DATASET/{file}')
    # print(our_fasttext_simple[['category', 'prediction']])
    icorrect = our_fasttext_simple.loc[our_fasttext_simple['category'] != our_fasttext_simple['prediction']]
    icorrect['file'] = file
    # return list(set(icorrect.url.to_list()))
    return icorrect

# our_fasttext_bi_lstm = get_incorrect_news_urls('our_fasttext_bi_lstm.csv')
# our_fasttext_simple = get_incorrect_news_urls('our_fasttext_simple.csv')
# our_keras_bi_gru = get_incorrect_news_urls('our_keras_bi_gru.csv')
# our_keras_cnn_bi_lstm = get_incorrect_news_urls('our_keras_cnn_bi_lstm.csv')
# our_random_forest_tfidf_ml_model = get_incorrect_news_urls('our_random_forest_tfidf_ml_model.csv')
# data = pd.concat([our_fasttext_bi_lstm, our_fasttext_simple, our_keras_bi_gru, our_keras_cnn_bi_lstm, our_random_forest_tfidf_ml_model])
# print(data)
#
# # print(our_fasttext_bi_lstm[['url', 'file']].groupby(by=["url"]).agg(['count']))
# # our_fasttext_bi_lstm['count'] = our_fasttext_bi_lstm.groupby(by=["url"]).transform('count')
# url_count = data['url'].value_counts().to_dict()
# with open('DATASET/url_count_incorrect.json', 'w') as file:
#     json.dump(url_count, file, indent=2)

# with open('DATASET/url_count_incorrect.json') as file:
#     url_count = json.load( file)
# urls = []
# for url in url_count:
#     if url_count[url] > 1:
#         print(url)
#         print(url_count[url])
#         urls.append(url)
# print(len(urls))
# print(len(url_count))
# with open('DATASET/urls_incorrect.json', 'w') as file:
#     json.dump(urls, file, indent=2)

with open('DATASET/urls_incorrect.json') as file:
    urls_incorrect = json.load( file)
print(len(urls_incorrect))
preprocessor = PreProcessor()
df = preprocessor.read_collected_data_incorrect_pred_removed(is_split=False)
print(len(df))
df = df[~df['url'].isin(urls_incorrect)]
print(len(df))
# df.to_csv(f'DATASET/collected_removed_urls_incorrect.csv')