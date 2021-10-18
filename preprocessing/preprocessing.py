from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import pickle
import dill
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from settings import DIR_DATASET, DIR_RESOURCES

class PreProcessor():

    def cleaning_documents(self, articles):
        news = articles.replace('\n',' ')
        # remove unnecessary punctuation
        news = re.sub('[^\u0980-\u09FF]',' ',str(news))
        # remove stopwords
        stp = open(DIR_RESOURCES + '/bangla_stopwords.txt','r',encoding='utf-8').read().split()
        result = news.split()
        news = [word.strip() for word in result if word not in stp ]
        news =" ".join(news)
        return news

    def read_data(self):
       data = pd.read_csv(DIR_DATASET + '/BanglaMCT7/train.csv')
       data_test = pd.read_csv(DIR_DATASET + '/BanglaMCT7/test.csv')

       # data, data_test = data.sample(100), data_test.sample(100)
       # data, data_test = data.reset_index(), data_test.reset_index()

       # Remove null
       print(f'Before removing null : {len(data)}')
       print(f'Before removing null : {len(data_test)}')
       data.dropna(inplace=True)
       data_test.dropna(inplace=True)
       print(f'After removing null : {len(data)}')
       print(f'After removing null : {len(data_test)}')

       # Remove duplicates
       print(f'Before removing duplicates : {len(data)}')
       print(f'Before removing duplicates : {len(data_test)}')
       data = data.drop_duplicates()
       data_test = data_test.drop_duplicates()
       print(f'After removing duplicates : {len(data)}')
       print(f'After removing duplicates : {len(data_test)}')

       data = data[['cleanText','category']]
       data_test = data_test[['cleanText','category']]

       # remove unnecessary punctuation & stopwords
       data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
       data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

       self.data, self.data_test = data, data_test

       return data, data_test

    def vectorize_tfidf(self, article, gram, name):
        tfidf = TfidfVectorizer(ngram_range=gram, use_idf=True, tokenizer=lambda x: x.split())
        x = tfidf.fit_transform(article)
        # save the label encoder into a pickle file
        # with open(DIR_RESOURCES + '/label_encoder.pickle', 'wb') as handle:
        with open(f'{DIR_RESOURCES}/{name}_tfidf_encoder.pickle', 'wb') as handle:
            dill.dump(tfidf, handle)
        return x

    def encode_category(self, category_col):
        with open(DIR_RESOURCES+'/label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        encoded_labels = le.transform(category_col)
        labels = np.array(encoded_labels)
        class_names = le.classes_
        return labels, class_names

    def handle_low_length_doc(self, data):
        # Calculate the Length of each Document
        data['Length'] = data.cleaned.apply(lambda x: len(x.split()))

        # Remove the Documents with least words
        dataset = data.loc[data.Length > 20]
        dataset = dataset.reset_index(drop=True)
        print("After Cleaning:", "\nRemoved {} Small Documents".format(len(data) - len(dataset)),
              "\nTotal Remaining Documents:", len(dataset))

        return data

    def preprocess_and_encode_data(self, data, is_test=True):
        if not is_test:
            data = self.handle_low_length_doc(data)
        num_words = 5000
        corpus, labels, class_names = self.encoded_texts_with_keras_tokenaizer(data, 300, num_words, is_test)
        print("\nShape of Encoded Corpus =====>", corpus.shape)
        print("\nShape of Encoded Corpus =====>", corpus.shape)
        return corpus, labels, class_names


    def tokenizer_info(self, mylist, bool):
        ordered = sorted(mylist.items(), key=lambda item: item[1], reverse=bool)
        for w, c in ordered[:10]:
            print(w, "\t", c)

    def encoded_texts_with_keras_tokenaizer(self, dataset, padding_length, max_words, is_test):

        if is_test:
            with open(DIR_RESOURCES+'/label_encoder.pickle', 'rb') as handle:
                le = pickle.load(handle)
            encoded_labels = le.transform(dataset.category)
            labels = np.array(encoded_labels)
            class_names = le.classes_
            with open(DIR_RESOURCES+'/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            sequences = tokenizer.texts_to_sequences(dataset.cleaned)
            corpus = pad_sequences(sequences, value=0.0, padding='post', maxlen=300)
        else:
            tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n-',
                                  split=' ', char_level=False, oov_token='<oov>', document_count=0)
            # Fit the tokenizer
            tokenizer.fit_on_texts(dataset.cleaned)
            # ============================== Tokenizer Info =================================
            (word_counts, word_docs, word_index, document_count) = (tokenizer.word_counts,
                                                                    tokenizer.word_docs,
                                                                    tokenizer.word_index,
                                                                    tokenizer.document_count)



            # =============================== Print all the information =========================
            print("\t\t\t====== Tokenizer Info ======")
            print("Words --> Counts:")
            self.tokenizer_info(word_counts, bool=True)
            print("\nWords --> Documents:")
            self.tokenizer_info(word_docs, bool=True)
            print("\nWords --> Index:")
            self.tokenizer_info(word_index, bool=True)
            print("\nTotal Documents -->", document_count)

            # =========================== Convert string into list of integer indices =================
            sequences = tokenizer.texts_to_sequences(dataset.cleaned)
            word_index = tokenizer.word_index
            print("\n\t\t\t====== Encoded Sequences ======",
                  "\nFound {} unique tokens".format(len(word_index)))
            print(dataset.cleaned[10], "\n", sequences[10])

            # ==================================== Pad Sequences ==============================
            corpus = keras.preprocessing.sequence.pad_sequences(sequences, value=0.0,
                                                                padding='post', maxlen=padding_length)
            print("\n\t\t\t====== Paded Sequences ======\n", dataset.cleaned[10], "\n", corpus[10])

            # save the tokenizer into a pickle file
            with open(DIR_RESOURCES + '/tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            le = LabelEncoder()
            le.fit(dataset.category)
            encoded_labels = le.transform(dataset.category)
            labels = np.array(encoded_labels)
            class_names = le.classes_

            # save the label encoder into a pickle file
            with open(DIR_RESOURCES + '/label_encoder.pickle', 'wb') as handle:
                pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)



        return corpus, labels, class_names
