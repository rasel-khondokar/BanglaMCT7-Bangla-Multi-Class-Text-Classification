import os

import dill
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Embedding, TextVectorization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from tensorflow import keras
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Reshape, Conv2D

from evaluation.performance import Evaluator
from preprocessing.preprocessing import PreProcessor
from settings import DIR_RESOURCES, DIR_IMAGES_EDA, DIR_IMAGES_HISTORY, DIR_BASE, MODEL_FASTTEXT_SIMPLE, \
    MODEL_FASTTEXT_DEEP_ANN


class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      accuracy_threshold = 0.99
      if(logs.get('val_accuracy')>accuracy_threshold):
        print("\nReached %2.2f%% accuracy so we will stop trianing" % (accuracy_threshold*100))
        self.model.stop_training = True

class ModelTrainer():

    def __init__(self, name, data, data_test):
        self.name = name
        self.data = data
        self.data_test = data_test

    def plot_accuracy_and_loss(self, name, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'{DIR_IMAGES_HISTORY}/{self.name}_{name}_accuracy.png')
        plt.close()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'{DIR_IMAGES_HISTORY}/{self.name}_{name}_loss.png')
        plt.close()

    def model_cnn_bi_lstm(self, num_classes, vocab_size, embedding_dimension, input_length):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dimension, input_length=input_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Dense(28, activation='relu'),
            tf.keras.layers.Dense(14, activation='relu'),
            keras.layers.Flatten(),
            tf.keras.layers.Dense(self, num_classes, activation='softmax')])
        return model

    def model_bi_gru(self, num_classes, vocab_size, embedding_dimension, input_length):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dimension, input_length=input_length),
            tf.keras.layers.Bidirectional(GRU(64, dropout=0.2)),
            tf.keras.layers.Dense(24, activation='relu'),
            keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')])
        return model

    def model_vdcnn_emb(self, embedding_layer, num_classes):
        sequence_length = 2048
        EMBEDDING_DIM = 300
        int_sequences_input = keras.Input(shape=(sequence_length,), dtype="int64")
        embedded_sequences = embedding_layer(int_sequences_input)
        reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedded_sequences)
        x = Conv2D(8, (3, 3), name='conv_1')(reshape)
        x = Conv2D(16, (5, 5), name='conv_2')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = Conv2D(32, (5, 5), name='conv_3')(x)
        x = Conv2D(64, (7, 7), name='conv_4')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
        x = Conv2D(128, (7, 7), name='conv_5')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(strides=(2, 2))(x)
        x = layers.Dense(2048)(x)
        x = layers.Dense(256)(x)
        preds = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(int_sequences_input, preds)
        return model

    def model_vdcnn(self, num_classes):
        inputs = keras.Input(shape=(784,))
        x = Conv2D(8, (3, 3), name='conv_1')(inputs)
        x = Conv2D(16, (5, 5), name='conv_2')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(32, (5, 5), name='conv_3')(x)
        x = Conv2D(64, (7, 7), name='conv_4')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
        x = Conv2D(128, (7, 7), name='conv_5')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = layers.MaxPooling2D(strides=(2, 2))(x)
        x = layers.Dense(2048)(x)
        x = layers.Dense(256)(x)
        preds = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, preds)
        return model

    def train_keras_tokenaizer(self, name):

        name = f'{self.name}_{name}_keras_tokenaizer'

        data, data_test = self.data, self.data_test

        preprocessor = PreProcessor()
        corpus, labels, class_names = preprocessor.preprocess_and_encode_data(data, is_test=False)

        X_train, X_valid, y_train, y_valid = train_test_split(corpus,labels, train_size=0.7,
                                                              test_size=0.1, random_state=0)

        embedding_dimension = 128
        input_length = 300
        vocab_size = 5000
        num_classes = len(list(np.unique(data['category'])))
        batch_size = 64
        num_epochs = 100
        filepath_best_model = f"{DIR_RESOURCES}/{name}_best_model.pkl"

        acc_callback = myCallback()



        # Saved the Best Model
        checkpoint = keras.callbacks.ModelCheckpoint(filepath_best_model, monitor='val_accuracy', verbose=2, save_best_only=True,
                                                     save_weights_only=False, mode='max')
        # callback list
        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callback_list = [acc_callback, checkpoint, stopping]

        if name == 'cnn_bi_lstm':
            model = self.model_cnn_bi_lstm(num_classes, vocab_size, embedding_dimension, input_length)
        else:
            model = self.model_bi_gru(num_classes, vocab_size, embedding_dimension, input_length)

        print(model.summary())
        model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train,
                            y_train,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=(X_valid, y_valid),
                            verbose=1,
                            callbacks = callback_list)

        self.plot_accuracy_and_loss(name, history)
        model_filepath = f'{DIR_RESOURCES}/{name}_document_categorization.pkl'
        model.save(model_filepath)

        evaluator = Evaluator(name, data, data_test)
        evaluator.evaluate_dl_model(filepath_best_model)

    def train_glove_tokenaizer(self, name):

        name = f'{self.name}_{name}_glove_tokenaizer'

        data, data_test = self.data, self.data_test
        corpus = data['cleaned']

        preprocessor = PreProcessor()
        labels, class_names = preprocessor.encode_category(self.data.category, name=f'{self.name}_')

        X_train, X_valid, y_train, y_valid = train_test_split(corpus, labels, train_size=0.7,
                                                              test_size=0.1, random_state=0)

        embedding_dimension = 300
        input_length = 300
        vocab_size = 5000
        num_classes = len(list(np.unique(data['category'])))
        batch_size = 64
        num_epochs = 100
        filepath_best_model = f"{DIR_RESOURCES}/{name}_best_model.pkl"

        acc_callback = myCallback()



        # Saved the Best Model
        checkpoint = keras.callbacks.ModelCheckpoint(filepath_best_model, monitor='val_accuracy', verbose=2, save_best_only=True,
                                                     save_weights_only=False, mode='max')
        # callback list
        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callback_list = [acc_callback, checkpoint, stopping]

        path_to_glove_file = "model_glove_word2vec_format_300.txt"

        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print("Found %s word vectors." % len(embeddings_index))
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=300)
        text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
        vectorizer.adapt(text_ds)
        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))
        num_tokens = len(voc) + 2
        embedding_dim = 300
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        embedding_layer = Embedding(
            num_tokens,
            embedding_dimension,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

        model = self.model_vdcnn_emb(embedding_layer, num_classes)

        print(model.summary())

        model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
        X_valid = vectorizer(np.array([[s] for s in X_valid])).numpy()
        history = model.fit(X_train,
                            y_train,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=(X_valid, y_valid),
                            verbose=1,
                            callbacks = callback_list)

        self.plot_accuracy_and_loss(name, history)
        model_filepath = f'{DIR_RESOURCES}/{name}_document_categorization.pkl'
        model.save(model_filepath)

        evaluator = Evaluator(name, data, data_test)
        evaluator.evaluate_dl_model(filepath_best_model)


    def model_fasttext_simple(self, num_classes, embedding_layer):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[], dtype=tf.string),
            embedding_layer,
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation='softmax')])
        return model

    def model_fasttext_deep_ann(self, num_classes, embedding_layer):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[], dtype=tf.string),
            embedding_layer,
            tf.keras.layers.Dense(224, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(28, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation='softmax')])
        return model

    def train_fasttext(self, model_name):

        model_name = f'{self.name}_{model_name}_fasttext'
        num_epochs = 100
        batch_size = 256

        data, data_test = self.data, self.data_test
        corpus = data['cleaned']

        preprocessor = PreProcessor()
        labels, class_names = preprocessor.encode_category(self.data.category, name=f'{self.name}_')

        X_train, X_valid, y_train, y_valid = train_test_split(corpus, labels, train_size=0.7,
                                                              test_size=0.1, random_state=0)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        num_classes = len(list(np.unique(data['category'])))

        filepath_best_model = f'{DIR_RESOURCES}/{model_name}_best_model.pkl'

        path_pretrained_model = 'text_module'
        embedding_layer = hub.KerasLayer(path_pretrained_model, trainable=False)

        acc_callback = myCallback()
        # Saved the Best Model
        checkpoint = keras.callbacks.ModelCheckpoint(filepath_best_model, monitor='val_accuracy', verbose=2, save_best_only=True,
                                                     save_weights_only=False, mode='max')
        # callback list
        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callback_list = [acc_callback, checkpoint, stopping]

        if model_name == f'{self.name}_{MODEL_FASTTEXT_SIMPLE}_fasttext':
            model = self.model_fasttext_simple(num_classes, embedding_layer)
        elif model_name == f'{self.name}_{MODEL_FASTTEXT_DEEP_ANN}_fasttext':
            model = self.model_fasttext_deep_ann(num_classes, embedding_layer)

        print(model.summary())
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer="adam", metrics=['accuracy'])

        history = model.fit(train_ds,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=test_ds,
                            verbose=1,
                            callbacks=callback_list)

        # save the model
        model_filepath = f'{DIR_RESOURCES}/{model_name}_document_categorization.pkl'
        model.save(model_filepath)

        self.plot_accuracy_and_loss(model_name, history)

        evaluator = Evaluator(model_name, data, data_test)
        evaluator.evaluate_dl_model(filepath_best_model)

    def train_tfidf_ml(self, name):
        name = f'{self.name}_{name}_tfidf_ml'

        data, data_test = self.data, self.data_test

        preprocessor = PreProcessor()

        corpus =  preprocessor.vectorize_tfidf(data.cleaned, (1,1), name)
        labels, class_names = preprocessor.encode_category(data.category)

        X_train, X_valid, y_train, y_valid = train_test_split(corpus, labels, train_size=0.7,
                                                              test_size=0.1, random_state=0)


        # model = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.05, min_samples_leaf=19, min_samples_split=3, n_estimators=100)
        model = LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.001)
        model.fit(X_train,y_train)
        model_filepath = f'{DIR_RESOURCES}/{name}_model.pickle'

        with open(model_filepath, 'wb') as handle:
            dill.dump(model, handle)

        y_pred = model.predict(X_valid)
        print(f"Accuracy score validation data : {accuracy_score(y_valid, y_pred)}")

        evaluator = Evaluator(name, data, data_test)
        evaluator.evaluate_ml_model(model_filepath)

