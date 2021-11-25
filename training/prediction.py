import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing.preprocessing import PreProcessor
from settings import DIR_RESOURCES, MODEL_FASTTEXT_SIMPLE

def get_category(text):
    preprocessor = PreProcessor()
    data_test =  pd.DataFrame([{'cleaned':text}])
    data_test['cleaned'] = data_test['cleaned'].apply(preprocessor.cleaning_documents)
    # Load the saved model
    model_filepath = f'{DIR_RESOURCES}/{MODEL_FASTTEXT_SIMPLE}_fasttext_best_model.pkl'
    model = tf.keras.models.load_model(model_filepath)
    prediction = model.predict(data_test)
    prediction = np.argmax(prediction, axis=1)
    category, class_names = preprocessor.decode_category(prediction)
    return category[0].upper()