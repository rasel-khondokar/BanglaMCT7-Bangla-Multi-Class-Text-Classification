import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
#Import module to split the datasets
from sklearn.model_selection import train_test_split
# Import modules to evaluate the metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, classification_report
from tensorflow import keras

from preprocessing.preprocessing import PreProcessor
import numpy as np  # array handling
from keras.datasets import imdb

# get cleaned train and test data
from training.training import ModelTrainer

preprocessor = PreProcessor()
data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()
train_data = data.cleaned.tolist()
train_labels = data.category.tolist()
test_data = data_test.cleaned.tolist()
test_labels = data_test.category.tolist()

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=None)


root_folder='.'
data_folder_name='data'
glove_filename= 'data/model_glove_word2vec_format_300.bin'

train_filename='train.csv'
# Variable for data directory
DATA_PATH = os.path.abspath(os.path.join(root_folder, data_folder_name))
glove_path = os.path.abspath(os.path.join(DATA_PATH, glove_filename))

# Both train and test set are in the root data directory
train_path = DATA_PATH
test_path = DATA_PATH

#Relevant columns
TEXT_COLUMN = 'text'
TARGET_COLUMN = 'target'

# We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format
# that will be loaded in the next section
from gensim.scripts.glove2word2vec import glove2word2vec

#glove_input_file = glove_filename
word2vec_output_file = glove_filename+'.word2vec'
glove2word2vec(glove_path, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
word2vec_output_file = glove_filename+'.word2vec'
model = KeyedVectors.load_word2vec_format('model_glove_word2vec_format_300.txt', binary=False)
from sklearn.decomposition import IncrementalPCA  # inital reduction
from sklearn.manifold import TSNE  # final reduction

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#
# review = [reverse_word_index.get(i-3, "") for i in train_data[0]]
# X_train=[]
# for doc in train_data:
#     X_train.append(' '.join([reverse_word_index.get(i - 3, "") for i in doc]))
#
# X_test=[]
# for doc in test_data:
#     X_test.append(' '.join([reverse_word_index.get(i - 3, "") for i in doc]))
X_train = train_data
X_test = test_data
print(len(X_train),len(X_test))
print(X_train[0])

class Word2VecVectorizer:
  def __init__(self, model):
    print("Loading in word vectors...")
    self.word_vectors = model
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    v = self.word_vectors.get_vector('হয়েছে')
    self.D = v.shape[0]

    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          # throws KeyError if word not found
          vec = self.word_vectors.get_vector(word)
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)

# Set a word vectorizer
vectorizer = Word2VecVectorizer(model)
# Get the sentence embeddings for the train dataset
Xtrain = vectorizer.fit_transform(X_train)
# Ytrain = train_labels
Ytrain = preprocessor.encode_category(pd.Series(train_labels))
# Get the sentence embeddings for the test dataset
Xtest = vectorizer.transform(X_test)
# Ytest = test_labels
Ytest =  preprocessor.encode_category(pd.Series(test_labels))
print(Xtrain.shape,Xtest.shape)
# from sklearn.ensemble import RandomForestClassifier
# # create the model, train it, print scores
# clf = RandomForestClassifier(n_estimators=3)
# clf.fit(Xtrain, Ytrain)
# print("train score:", clf.score(Xtrain, Ytrain))
# print("test score:", clf.score(Xtest, Ytest))
# Ypred = clf.predict(Xtest)
# print(classification_report(Ytest, Ypred))
trainer = ModelTrainer('vdcnn', data, data_test)
model = trainer.model_vdcnn(7)
print(model.summary())
model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(Xtrain,
                    Ytrain,
                    epochs=2,
                    batch_size=64,
                    validation_data=(Xtest, Ytest),
                    verbose=1)