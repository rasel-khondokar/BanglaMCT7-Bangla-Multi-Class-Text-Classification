import os
import pickle

import dill
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from bert import run_bert_test
from evaluation.performance import Evaluator
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_FASTTEXT_SIMPLE, DIR_RESOURCES, MODEL_BIDIRECTIONAL_GRU, MODEL_CNN_BIDIRECTIONAL_LSTM, \
    MODEL_FASTTEXT_DEEP_ANN, MODEL_ML, DIR_PERFORMENCE_REPORT, MODEL_BERT_MULTILANGUAL_CASED, MODEL_BERT_CESBUETNLP, \
    MODEL_BERT_MONSOON_NLP, MODEL_BERT_SAGORSARKAR

preprocessor = PreProcessor()

# datasets = {'osac_rm_oth':preprocessor.read_osac(is_split=False),
#             'prothomalo_rm_oth':preprocessor.read_prothomalo(is_split=False),
#             'bard_rm_oth':preprocessor.read_bard(is_split=False)
# }

df, df_test = preprocessor.read_collected_data_incorrect_pred_removed()
datasets = {'our_rm_oth':df_test}


def run_dl_test():
    models = {'incorrect_keras_bi_gru': ['incorrect_pred_removed_bi_gru_keras_tokenaizer_best_model.pkl',
                                         MODEL_BIDIRECTIONAL_GRU],
              'incorrect_keras_cnn_bi_lstm': ['incorrect_pred_removed_cnn_bi_lstm_keras_tokenaizer_best_model.pkl',
                                              MODEL_CNN_BIDIRECTIONAL_LSTM],
              'incorrect_fasttext_simple': ['incorrect_pred_removed_simple_fasttext_best_model.pkl',
                                            MODEL_FASTTEXT_SIMPLE],
              'incorrect_fasttext_bi_lstm': ['incorrect_pred_removed_bi_lstm_fasttext_best_model.pkl',
                                             MODEL_FASTTEXT_DEEP_ANN],
              'incorrect_random_forest_tfidf_ml_model': ['random_forest_tfidf_ml_model.pickle', MODEL_ML]
              }

    for model_key in models:
        for dataset_name in datasets:
            dataset = datasets[dataset_name]
            model_name = models[model_key][1]

            with open(DIR_RESOURCES + f'/collected_label_encoder.pickle', 'rb') as handle:
                le = pickle.load(handle)
            encoded_labels = le.transform(dataset['category'])
            labels = np.array(encoded_labels)

            # Load the saved model
            filepath_best_model = f"{DIR_RESOURCES}/{models[model_key][0]}"

            if 'tfidf' in model_key:
                with open(filepath_best_model, 'rb') as handle:
                    model = dill.load(handle)
            else:
                model = tf.keras.models.load_model(filepath_best_model)

            if "fasttext" in model_key:
                x = dataset['cleaned']
            elif 'tfidf' in model_key:
                with open(f'{DIR_RESOURCES}/random_forest_tfidf_ml_tfidf_encoder.pickle', 'rb') as handle:
                    tfidf = dill.load(handle)
                x = tfidf.transform(dataset['cleaned'])
            else:
                x, labels, class_names = preprocessor.preprocess_and_encode_data(dataset, is_test=True)

            predictions = model.predict(x)
            if not 'tfidf' in model_key:
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_pred = predictions

            print(le.classes_)
            decoded_labels = np.array(le.inverse_transform(y_pred))
            dataset['prediction'] = decoded_labels
            # dataset.to_csv(f'DATASET/{dataset_name}_{model_key}.csv')
            print(set(decoded_labels))
            print(set(dataset['category']))
            report = classification_report(dataset['category'], decoded_labels)

            name = 'test'
            with open(f'{DIR_PERFORMENCE_REPORT}/{name}_{dataset_name}_{model_key}.txt', 'w') as file:
                file.write(str(report))
            cm = confusion_matrix(dataset['category'], decoded_labels)
            ConfusionMatrixDisplay.from_predictions(dataset['category'], decoded_labels, xticks_rotation=18.0, cmap='YlGn')
            plt.savefig(f'{DIR_PERFORMENCE_REPORT}/{name}_{dataset_name}_{model_key}.png')


def run_test_othres_on_bert():
    # models = {'incorrect_keras_bi_gru': ['incorrect_pred_removed_bi_gru_keras_tokenaizer_best_model.pkl',
    #                                      MODEL_BIDIRECTIONAL_GRU],
    #           'incorrect_keras_cnn_bi_lstm': ['incorrect_pred_removed_cnn_bi_lstm_keras_tokenaizer_best_model.pkl',
    #                                           MODEL_CNN_BIDIRECTIONAL_LSTM],
    #           'incorrect_fasttext_simple': ['incorrect_pred_removed_simple_fasttext_best_model.pkl',
    #                                         MODEL_FASTTEXT_SIMPLE],
    #           'incorrect_fasttext_bi_lstm': ['incorrect_pred_removed_bi_lstm_fasttext_best_model.pkl',
    #                                          MODEL_FASTTEXT_DEEP_ANN],
    #           'incorrect_random_forest_tfidf_ml_model': ['random_forest_tfidf_ml_model.pickle', MODEL_ML]
    #           }
    models = {f'incorrect_{MODEL_BERT_MULTILANGUAL_CASED}': [MODEL_BERT_MULTILANGUAL_CASED.replace("/", "_"),
                                         MODEL_BERT_MULTILANGUAL_CASED],
              f'incorrect_{MODEL_BERT_CESBUETNLP}': [MODEL_BERT_CESBUETNLP.replace("/", "_"),
                                                             MODEL_BERT_CESBUETNLP],
              f'incorrect_{MODEL_BERT_MONSOON_NLP}': [MODEL_BERT_MONSOON_NLP.replace("/", "_"),
                                                             MODEL_BERT_MONSOON_NLP],
              f'incorrect_{MODEL_BERT_SAGORSARKAR}': [MODEL_BERT_SAGORSARKAR.replace("/", "_"),
                                                     MODEL_BERT_SAGORSARKAR]
              }

    for model_key in models:
        for dataset_name in datasets:
            dataset = datasets[dataset_name]
            model_name = models[model_key][1]
            run_bert_test(model_name.replace("/", "_"), dataset, is_test=False, report_name='test_')

def main():
    # run_dl_test()
    run_test_othres_on_bert()

if __name__=='__main__':
    main()