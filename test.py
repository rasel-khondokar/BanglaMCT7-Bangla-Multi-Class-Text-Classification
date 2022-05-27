
# Train and evaluation
from training.training import ModelTrainer

trainer = ModelTrainer('name', [], [])
embedding_dimension = 128
input_length = 300
vocab_size = 5000
batch_size = 64
num_epochs = 100
model = trainer.model_cnn_bi_lstm(7, vocab_size, embedding_dimension, input_length)
print(model.summary())
# import json
#
# from preprocessing.preprocessing import PreProcessor
#
# preprocessor = PreProcessor()
# data, data_test = preprocessor.read_cyberbullying_dataset()
#
# '''
# Before removing null Train data : 35932
# After removing null Train data : 35191
# Before removing null Test data: 8984
# After removing null Test data : 8807
# '''

# import json
# import re
#
# import pandas as pd
# from configs import BASE_DIR
# dataset_dir = f'{BASE_DIR}/DATASET/'
# file = f'{dataset_dir}scraped_data.json'
# data = json.load(open(f'{file}'))
# df = pd.DataFrame()
# for d in data:
#     df = df.append(d['comments'])
# def clean_text(text):
#     if isinstance(text, str):
#         # remove non bangla text
#         text = "".join(i for i in text if i in [".", "।"] or 2432 <= ord(i) <= 2559 or ord(i) == 32)
#         # remove space
#         text = text.replace('\n', ' ')
#         # remove unnecessary punctuation
#         text = re.sub('[^\u0980-\u09FF]', ' ', str(text))
#         text = " ".join(text.split())
#         return text
# df.comment_text = df.comment_text.apply(clean_text)
# comments = df.comment_text.to_list()
# comments = [i for i in comments if i]

# cols = df.columns
# comments = []
# for i in range(10):
#     for col in cols:
#         if 'comment_text' in col:
#             if pd.notnull(df.loc[i][col]):
#                 comments.append(df.loc[i][col])
# json.dump(comments, open('DATASET/scraped_data.json', 'w'), indent=2, ensure_ascii=False)
# d = pd.DataFrame(comments, columns=['comments'])
# d.to_csv('DATASET/comments.csv', index=False)
# df = df[df['label'].isin(['threat', 'troll'])]
# print(df[['comment', 'label']].head())

from bert import test_bert_model, run_bert_test
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_BERT_MULTILANGUAL_CASED, DIR_RESOURCES

models = {f'incorrect_{MODEL_BERT_MULTILANGUAL_CASED}': [MODEL_BERT_MULTILANGUAL_CASED.replace("/", "_"),
                                                         MODEL_BERT_MULTILANGUAL_CASED]
          }
preprocessor = PreProcessor()
df, df_test = preprocessor.read_collected_data_incorrect_pred_removed()
run_bert_test(MODEL_BERT_MULTILANGUAL_CASED, df_test, is_test=False)

