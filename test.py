from bert import test_bert_model, run_bert_test
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_BERT_MULTILANGUAL_CASED, DIR_RESOURCES

models = {f'incorrect_{MODEL_BERT_MULTILANGUAL_CASED}': [MODEL_BERT_MULTILANGUAL_CASED.replace("/", "_"),
                                                         MODEL_BERT_MULTILANGUAL_CASED]
          }
preprocessor = PreProcessor()
df, df_test = preprocessor.read_collected_data_incorrect_pred_removed()
run_bert_test(MODEL_BERT_MULTILANGUAL_CASED, df_test, is_test=False)