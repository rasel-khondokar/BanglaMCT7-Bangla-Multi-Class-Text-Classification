import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_FASTTEXT_SIMPLE, DIR_RESOURCES

preprocessor = PreProcessor()

dataset = preprocessor.read_osac(is_split=False)

model_name = MODEL_FASTTEXT_SIMPLE
name = 'collected'
model_name = f'{name}_{model_name}_fasttext'

with open(DIR_RESOURCES + f'/collected_label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)
encoded_labels = le.transform(dataset['category'])
labels = np.array(encoded_labels)

# Load the saved model
filepath_best_model = f"{DIR_RESOURCES}/collected_simple_fasttext_best_model.pkl"
model = tf.keras.models.load_model(filepath_best_model)
predictions = model.predict(dataset['cleaned'])
y_pred = np.argmax(predictions, axis=1)
print(le.classes_)
decoded_labels = np.array(le.inverse_transform(y_pred))
print(set(decoded_labels))
print(set(dataset['category']))
report = classification_report(dataset['category'], decoded_labels)
print(report)
cm = confusion_matrix(dataset['category'], decoded_labels)
print(cm)