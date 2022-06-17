import pandas as pd
from settings import DIR_PERFORMENCE_REPORT


def make_df_from_cls_report(filename):
    with open(f'{DIR_PERFORMENCE_REPORT}{filename}.txt') as file:
        text = file.read()
    text = text.split('\n')[12:20]
    data = []
    for i in text:
        d = [j for j in i.split(' ') if j]
        data.append(d)
    data = pd.DataFrame(data, columns=['class', 'precision', 'recall', 'f1-score', 'support'])
    data.iloc[0, 0] = filename[30:]
    return data

def make_excel_report_from_cls_report(name, files):
    df = pd.DataFrame()
    for f in files:
        df = df.append(make_df_from_cls_report(f), ignore_index=True)
    df.to_excel(f'{DIR_PERFORMENCE_REPORT}performence table - {name}.xlsx')
make_excel_report_from_cls_report('prothomalo', [
    'test_others_prothomalo_rm_oth_automl_best_model_LinearSVC',
    'test_others__prothomalo_rm_oth_incorrect_fasttext_bi_lstm',
    'test_others__prothomalo_rm_oth_incorrect_fasttext_simple',
    'test_others__prothomalo_rm_oth_incorrect_keras_cnn_bi_lstm',
    'test_others__prothomalo_rm_oth_incorrect_keras_bi_gru',
    'test_others_prothomalo_rm_oth_bert-base-multilingual-cased_train',
    'test_others_prothomalo_rm_oth_csebuetnlp_banglabert_train',
    'test_others_prothomalo_rm_oth_monsoon-nlp_bangla-electra_train',
    'test_others_prothomalo_rm_oth_sagorsarker_bangla-bert-base_train'
])