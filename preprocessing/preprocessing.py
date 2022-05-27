import json
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import pickle
import dill
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

from configs import BASE_DIR
from settings import DIR_DATASET, DIR_RESOURCES

class PreProcessor():

    def cleaning_documents(self, articles):
        # remove non bangla text
        news = "".join(i for i in articles if i in [".","।"] or 2432 <= ord(i) <= 2559 or ord(i)== 32)
        # remove space
        news = news.replace('\n',' ')
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

       data, data_test = data.sample(100), data_test.sample(100)
       data, data_test = data.reset_index(), data_test.reset_index()

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

    def read_osac(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/OSAC/category'
        cat_dirs = os.listdir(dataset_dir)
        df = pd.DataFrame()
        target_categories = ['economics', 'education','science_tech', 'politics', 'entertainment', 'international', 'sports']
        for dir in cat_dirs:
            if dir in target_categories:
                dir_cat = f'{dataset_dir}/{dir}/{dir}'
                files = os.listdir(dir_cat)
                for i, file in enumerate(files):
                    with open(f'{dir_cat}/{file}') as f:
                        text = f.read()
                        df = df.append({'cleanText':text, 'category':dir},
                                                 ignore_index=True)

                        # if i>10:
                        #     break

        df = df[df['category'].isin(target_categories)]
        df['category'] = df['category'].replace(target_categories,
                                                ['economy', 'education','technology', 'politics', 'entertainment',
                                                 'international', 'sports'])
        if is_split:
            data, data_test = train_test_split(df, test_size=.2)

            self.data, self.data_test = data, data_test

            print(f'Before removing null : {len(data)}')
            print(f'Before removing null : {len(data_test)}')
            data.dropna(inplace=True)
            data_test.dropna(inplace=True)
            print(f'After removing null : {len(data)}')
            print(f'After removing null : {len(data_test)}')

            # Remove duplicates
            print(f'Before removing duplicates : {len(data)}')
            print(f'Before removing duplicates : {len(data_test)}')
            # data = data.drop_duplicates(subset=['url'])
            # data_test = data_test.drop_duplicates(subset=['url'])
            print(f'After removing duplicates : {len(data)}')
            print(f'After removing duplicates : {len(data_test)}')

            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            return data, data_test
        else:
            print(f'Before removing null : {len(df)}')
            df.dropna(inplace=True)
            print(f'After removing null : {len(df)}')

            df = df[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def read_bard(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/BARD'
        cat_dirs = os.listdir(dataset_dir)
        df = pd.DataFrame()
        for dir in cat_dirs:
            dir_cat = f'{dataset_dir}/{dir}'
            files = os.listdir(dir_cat)
            for i, file in enumerate(files):
                with open(f'{dir_cat}/{file}') as f:
                    text = f.read()
                    df = df.append({'cleanText':text, 'category':dir},
                                             ignore_index=True)

                    # if i>100:
                    #     break

        df = df[df['category'].isin(['economy', 'entertainment', 'international', 'sports'])]

        if is_split:
            data, data_test = train_test_split(df, test_size=.2)

            self.data, self.data_test = data, data_test

            print(f'Before removing null : {len(data)}')
            print(f'Before removing null : {len(data_test)}')
            data.dropna(inplace=True)
            data_test.dropna(inplace=True)
            print(f'After removing null : {len(data)}')
            print(f'After removing null : {len(data_test)}')

            # Remove duplicates
            print(f'Before removing duplicates : {len(data)}')
            print(f'Before removing duplicates : {len(data_test)}')
            # data = data.drop_duplicates(subset=['url'])
            # data_test = data_test.drop_duplicates(subset=['url'])
            print(f'After removing duplicates : {len(data)}')
            print(f'After removing duplicates : {len(data_test)}')

            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            return data, data_test
        else:
            print(f'Before removing null : {len(df)}')
            df.dropna(inplace=True)
            print(f'After removing null : {len(df)}')

            df = df[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def read_prothomalo(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/prothomALo'
        files = os.listdir(dataset_dir)
        dataset = pd.DataFrame()
        for file in files:
            data_file = f'{dataset_dir}/{file}'
            df = pd.read_csv(data_file)
            dataset = dataset.append(df, ignore_index=True)

        # dataset = dataset.sample(100, random_state=0)

        df = pd.DataFrame()
        df['cleanText'] = dataset['content']
        df['category'] = dataset['section']
        df = df[df['category'].isin(['economy', 'education', 'entertainment', 'international', 'politics', 'sports', 'technology'])]

        if is_split:
            data, data_test = train_test_split(df, test_size=.01)

            self.data, self.data_test = data, data_test


            print(f'Before removing null : {len(data)}')
            print(f'Before removing null : {len(data_test)}')
            data.dropna(inplace=True)
            data_test.dropna(inplace=True)
            print(f'After removing null : {len(data)}')
            print(f'After removing null : {len(data_test)}')

            # Remove duplicates
            print(f'Before removing duplicates : {len(data)}')
            print(f'Before removing duplicates : {len(data_test)}')
            # data = data.drop_duplicates(subset=['url'])
            # data_test = data_test.drop_duplicates(subset=['url'])
            print(f'After removing duplicates : {len(data)}')
            print(f'After removing duplicates : {len(data_test)}')

            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            return data, data_test
        else:
            print(f'Before removing null : {len(df)}')
            df.dropna(inplace=True)
            print(f'After removing null : {len(df)}')

            df = df[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def read_collected_data(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/'
        file = f'{dataset_dir}collected_data.csv'
        dataset = pd.read_csv(file)
        dataset = dataset.append(pd.read_csv(f'{dataset_dir}collected_data_1.csv'), ignore_index=True)
        # dataset = dataset.sample(100)
        # dataset = dataset.reset_index()
        if is_split:
            data, data_test = train_test_split(dataset, test_size=.2, stratify=dataset.category.values)

            # Remove null
            print(f'Before removing null Train data : {len(data)}')
            data.dropna(inplace=True)
            print(f'After removing null Train data : {len(data)}')

            print(f'Before removing null Test data: {len(data_test)}')
            data_test.dropna(inplace=True)
            print(f'After removing null Test data : {len(data_test)}')

            # Remove duplicates
            print(f'Before removing duplicates Train data: {len(data)}')
            data = data.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Train data : {len(data)}')

            print(f'Before removing duplicates Test data : {len(data_test)}')
            data_test = data_test.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Test data : {len(data_test)}')

            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            self.data, self.data_test = data, data_test

            return data, data_test
        else:
            print(f'Before removing null : {len(dataset)}')
            dataset.dropna(inplace=True)
            print(f'After removing null : {len(dataset)}')
            # Remove duplicates
            print(f'Before removing duplicates Train data: {len(dataset)}')
            dataset = dataset.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Train data : {len(dataset)}')

            df = dataset[['url', 'cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def read_collected_data_incorrect_pred_removed(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/'
        file = f'{dataset_dir}collected_removed_urls_incorrect.csv'
        dataset = pd.read_csv(file)
        # dataset = dataset.sample(500)

        dataset = dataset.reset_index()
        if is_split:
            data, data_test = train_test_split(dataset, test_size=.2, stratify=dataset.category.values)

            # Remove null
            print(f'Before removing null Train data : {len(data)}')
            data.dropna(inplace=True)
            print(f'After removing null Train data : {len(data)}')

            print(f'Before removing null Test data: {len(data_test)}')
            data_test.dropna(inplace=True)
            print(f'After removing null Test data : {len(data_test)}')

            # Remove duplicates
            print(f'Before removing duplicates Train data: {len(data)}')
            data = data.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Train data : {len(data)}')

            print(f'Before removing duplicates Test data : {len(data_test)}')
            data_test = data_test.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Test data : {len(data_test)}')

            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            self.data, self.data_test = data, data_test

            return data, data_test
        else:
            print(f'Before removing null : {len(dataset)}')
            dataset.dropna(inplace=True)
            print(f'After removing null : {len(dataset)}')
            # Remove duplicates
            print(f'Before removing duplicates Train data: {len(dataset)}')
            dataset = dataset.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Train data : {len(dataset)}')

            df = dataset[['url', 'cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def read_cyberbullying_labeled_data(self):
        dataset_dir = f'{BASE_DIR}/DATASET/cyberbullying'
        files = os.listdir(dataset_dir)
        dataset = []
        for file in files:
            data_file = f'{dataset_dir}/{file}'
            if '.json' in data_file:
                with open(data_file) as file:
                    data = json.load(file)
                if data:
                    for i in data:
                        try:
                            d = {'comment' : i['data']['comments'],
                            'label' : i['annotations'][0]['result'][0]['value']['choices'][0]}
                            dataset.append(d)
                        except Exception as e:
                            print(e)

        return pd.DataFrame(dataset)

    def read_cyberbullying_dataset(self, is_split=True):
        dataset_dir = f'{BASE_DIR}/DATASET/'
        file = f'{dataset_dir}cyberbullying/cyberbullying.csv'
        dataset = pd.read_csv(file)
        # dataset = pd.DataFrame()
        dataset_labeled = self.read_cyberbullying_labeled_data()
        dataset = dataset.append(dataset_labeled, ignore_index=True)

        dataset = dataset.sample(100)

        dataset['cleanText'] = dataset['comment']
        dataset['category'] = dataset['label']
        dataset = dataset[['cleanText', 'category']]
        dataset = dataset.reset_index()
        if is_split:
            data, data_test = train_test_split(dataset, test_size=.2, stratify=dataset.category.values)

            # Remove null
            print(f'Before removing null Train data : {len(data)}')
            data.dropna(inplace=True)
            print(f'After removing null Train data : {len(data)}')

            print(f'Before removing null Test data: {len(data_test)}')
            data_test.dropna(inplace=True)
            print(f'After removing null Test data : {len(data_test)}')



            data = data[['cleanText', 'category']]
            data_test = data_test[['cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            data['cleaned'] = data['cleanText'].apply(self.cleaning_documents)
            data_test['cleaned'] = data_test['cleanText'].apply(self.cleaning_documents)

            self.data, self.data_test = data, data_test

            return data, data_test
        else:
            print(f'Before removing null : {len(dataset)}')
            dataset.dropna(inplace=True)
            print(f'After removing null : {len(dataset)}')
            # Remove duplicates
            print(f'Before removing duplicates Train data: {len(dataset)}')
            dataset = dataset.drop_duplicates(subset=['url'])
            print(f'After removing duplicates Train data : {len(dataset)}')

            df = dataset[['url', 'cleanText', 'category']]

            # remove unnecessary punctuation & stopwords
            df['cleaned'] = df['cleanText'].apply(self.cleaning_documents)

            return df

    def vectorize_tfidf(self, article, gram, name):
        tfidf = TfidfVectorizer(ngram_range=gram, use_idf=True, tokenizer=lambda x: x.split())
        x = tfidf.fit_transform(article)
        # save the label encoder into a pickle file
        # with open(DIR_RESOURCES + '/label_encoder.pickle', 'wb') as handle:
        with open(f'{DIR_RESOURCES}/{name}_tfidf_encoder.pickle', 'wb') as handle:
            dill.dump(tfidf, handle)
        return x

    def encode_category(self, category_col, is_test=False, name=''):
        if is_test:
            with open(DIR_RESOURCES+f'/{name}label_encoder.pickle', 'rb') as handle:
                le = pickle.load(handle)
            encoded_labels = le.transform(category_col)
            labels = np.array(encoded_labels)
            class_names = le.classes_
        else:
            le = LabelEncoder()
            le.fit(category_col)
            encoded_labels = le.transform(category_col)
            labels = np.array(encoded_labels)
            class_names = le.classes_

            # save the label encoder into a pickle file
            with open(DIR_RESOURCES + f'/{name}label_encoder.pickle', 'wb') as handle:
                pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return labels, class_names

    def decode_category(self, encoded_category, name=''):
        with open(DIR_RESOURCES+f'/{name}label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        decoded_labels = le.inverse_transform(encoded_category)
        labels = np.array(decoded_labels)
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
        # if not is_test:
            # data = self.handle_low_length_doc(data)
        num_words = 5000
        corpus, labels, class_names = self.encoded_texts_with_keras_tokenaizer(data, 300, num_words, is_test)
        print("\nShape of Encoded Corpus =====>", corpus.shape)
        print("\nShape of Encoded Corpus =====>", corpus.shape)
        return corpus, labels, class_names

    def preprocess_and_glove_encode_data(self, data, is_test=True):
        # if not is_test:
        #     data = self.handle_low_length_doc(data)
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
            try:
                print("\n\t\t\t====== Encoded Sequences ======",
                      "\nFound {} unique tokens".format(len(word_index)))
                print(dataset.cleaned[10], "\n", sequences[10])
            except Exception as e:
                print(f'Error during showing Encoded Sequences')

            # ==================================== Pad Sequences ==============================
            corpus = keras.preprocessing.sequence.pad_sequences(sequences, value=0.0,
                                                                padding='post', maxlen=padding_length)

            # print("\n\t\t\t====== Paded Sequences ======\n", dataset.cleaned[10], "\n", corpus[10])

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

    def encoded_texts_with_glove_tokenaizer(self, dataset, padding_length, max_words, is_test):

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
            try:
                print("\n\t\t\t====== Encoded Sequences ======",
                      "\nFound {} unique tokens".format(len(word_index)))
                print(dataset.cleaned[10], "\n", sequences[10])
            except Exception as e:
                print(f'Error during showing Encoded Sequences')

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
