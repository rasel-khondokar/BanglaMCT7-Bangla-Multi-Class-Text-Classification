import tensorflow as tf
import numpy as np
from time import time
n = 8192
dtype = tf.float32

def benchmark_processor():
    start_time = time()
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)
    finish_time = time()
    elapsed_time = finish_time - start_time
    return elapsed_time
tf.config.list_physical_devices()
with tf.device('/CPU:0'):
    cpu_time = benchmark_processor()
print(cpu_time)





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
d = pd.DataFrame(comments, columns=['comments'])
d.to_csv('DATASET/comments.csv', index=False)
# df = df[df['label'].isin(['threat', 'troll'])]
# print(df[['comment', 'label']].head())