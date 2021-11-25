import dill
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing.preprocessing import PreProcessor
from settings import DIR_RESOURCES, DIR_PERFORMENCE_REPORT


class Evaluator():
    def __init__(self, name, data, data_test):
        self.name = name
        self.data = data
        self.data_test = data_test

    def save_report(self, x, y, is_test=True, is_dl=True):
        if is_test:
            data_split = 'test'
        else:
            data_split = 'train'
        predictions = self.model.predict(x)

        if is_dl:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = predictions

        preprocessor = PreProcessor()
        y, class_names = preprocessor.decode_category(y)
        y_pred, class_names = preprocessor.decode_category(y_pred)

        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)
        print(report)
        report_filename =  self.model_filepath.replace(DIR_RESOURCES, '').replace('.pkl', '').replace('.pickle', '')
        with open(f'{DIR_PERFORMENCE_REPORT}{report_filename}_{data_split}.txt', 'w') as file:
            file.write('___________________ confusion_matrix _____________________\n')
            file.write(str(cm))
            file.write('\n\n\n')
            file.write('___________________ classification report _____________________\n')
            file.write(str(report))

    def evaluate_ml_model(self, model_filepath):
        self.model_filepath = model_filepath

        with open(f'{DIR_RESOURCES}/{self.name}_tfidf_encoder.pickle', 'rb') as handle:
            tfidf = dill.load(handle)

        X_train = tfidf.transform(self.data['cleaned'])
        X_test = tfidf.transform(self.data_test['cleaned'])

        preprocessor = PreProcessor()
        y_train, class_names_train =  preprocessor.encode_category(self.data.category)
        y_test, class_names_test =  preprocessor.encode_category(self.data_test.category)

        # Load the saved model
        with open(model_filepath, 'rb') as handle:
            self.model = dill.load(handle)



        self.save_report(X_train, y_train, is_test=False, is_dl=False)
        self.save_report(X_test, y_test, is_test=True, is_dl=False)

    def evaluate_dl_model(self, model_filepath):
        self.model_filepath = model_filepath
        preprocessor = PreProcessor()
        if "fasttext" in model_filepath:
            X_train, X_test = self.data['cleaned'], self.data_test['cleaned']

            y_train, class_names_train =  preprocessor.encode_category(self.data.category)
            y_test, class_names_test =  preprocessor.encode_category(self.data_test.category)
        else:
            X_train, y_train, class_names_train = preprocessor.preprocess_and_encode_data(self.data, is_test=True)
            X_test, y_test, class_names_test = preprocessor.preprocess_and_encode_data(self.data_test, is_test=True)

        # Load the saved model
        self.model = tf.keras.models.load_model(self.model_filepath)
        self.save_report(X_train, y_train, is_test=False)
        self.save_report(X_test, y_test, is_test=True)

