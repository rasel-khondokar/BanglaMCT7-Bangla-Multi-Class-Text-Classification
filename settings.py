import os

MODEL_BIDIRECTIONAL_GRU = "bi_gru"
MODEL_CNN_BIDIRECTIONAL_LSTM = "cnn_bi_lstm"
MODEL_FASTTEXT_DEEP_ANN = "bi_lstm"
MODEL_FASTTEXT_SIMPLE = "simple"
MODEL_ML = "random_forest"

MODEL_BERT_MULTILANGUAL_CASED = 'bert-base-multilingual-cased'
MODEL_BERT_CESBUETNLP = 'csebuetnlp/banglabert'
MODEL_BERT_MONSOON_NLP = 'monsoon-nlp/bangla-electra'
MODEL_BERT_SAGORSARKAR = 'sagorsarker/bangla-bert-base'

MODEL_SELECTED = MODEL_FASTTEXT_DEEP_ANN

DIR_BASE = f'{os.path.dirname(os.path.realpath(__file__))}/'
DIR_DATASET = 'DATASET/'
DIR_REPORT = 'REPORT/'
DIR_EDA = 'EDA/'
DIR_IMAGES_EDA = DIR_REPORT + 'IMAGES/EDA/'
DIR_IMAGES_HISTORY = DIR_REPORT + 'IMAGES/HISTORY/'
DIR_PERFORMENCE_REPORT = DIR_REPORT + 'PERFORMENCE/'
DIR_RESOURCES = DIR_BASE + 'resources/'


