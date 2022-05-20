import matplotlib
import numpy as np
import pandas as pd

from exploratory_analysis.eda import EDA
from preprocessing.preprocessing import PreProcessor
#
# preprocessor = PreProcessor()
# data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()
from scraping.helpers import make_dir_if_not_exists
from settings import DIR_REPORT, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_IMAGES_EDA


def main():
    for dir in [DIR_REPORT, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_IMAGES_EDA]:
        make_dir_if_not_exists(dir)
    name = 'incorrect_pred_removed'
    # get cleaned train and test data
    preprocessor = PreProcessor()
    data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()

    # exploratory data analysis
    try:
        eda = EDA(data, name)
        eda.visualize()
        eda.analyze()
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()

# class_distribution = data['category'].value_counts(sort=True, ascending=False)
# class_distribution = class_distribution.to_frame()
# class_distribution = class_distribution.reset_index()
# class_distribution.set_axis(['class', 'category'], axis=1, inplace=True)
# class_distribution.to_excel(f'REPORT/class_distribution.xlsx')
#
# def get_document_length_distribution(data):
#     data['Length'] = data.cleaned.apply(lambda x: len(x.split()))
#     matplotlib.rc_file_defaults()
#     frequency = dict()
#     for i in data.Length:
#         frequency[i] = frequency.get(i, 0) + 1
#     df = pd.DataFrame()
#     df['Length'] = frequency.keys()
#     df['Frequency'] = frequency.values()
#     return df
# document_length_distribution = get_document_length_distribution(data)
# document_length_distribution.to_excel(f'REPORT/document_length_distribution.xlsx')
#
# def get_data_summary(data):
#     documents = []
#     words = []
#     u_words = []
#     # find class names
#     class_label = [k for k, v in data.category.value_counts().to_dict().items()]
#     for label in class_label:
#         word_list = [word.strip().lower() for t in list(data[data.category == label].cleaned) for word in
#                      t.strip().split()]
#         counts = dict()
#         for word in word_list:
#             counts[word] = counts.get(word, 0) + 1
#         # sort the dictionary of word list
#         ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
#         # Documents per class
#         documents.append(len(list(data[data.category == label].cleaned)))
#         # Total Word per class
#         words.append(len(word_list))
#         # Unique words per class
#         u_words.append(len(np.unique(word_list)))
#
#         print("\nClass Name : ", label)
#         print("Number of Documents:{}".format(len(list(data[data.category == label].cleaned))))
#         print("Number of Words:{}".format(len(word_list)))
#         print("Number of Unique Words:{}".format(len(np.unique(word_list))))
#         print("Most Frequent Words:\n")
#         for k, v in ordered[:10]:
#             print("{}\t{}".format(k, v))
#
#     data_matrix = pd.DataFrame({'Total Documents': documents,
#                                 'Total Words': words,
#                                 'Unique Words': u_words,
#                                 'Class Names': class_label})
#     return data_matrix
#
# data_summary = get_data_summary(data)
# data_summary.to_excel(f'REPORT/data_summary.xlsx')
