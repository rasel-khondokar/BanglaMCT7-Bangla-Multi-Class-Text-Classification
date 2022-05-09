import sys
import warnings

from scraping.helpers import make_dir_if_not_exists
from test_all_dataset import run_automl_test

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from exploratory_analysis.eda import EDA
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_SELECTED, MODEL_BIDIRECTIONAL_GRU, MODEL_CNN_BIDIRECTIONAL_LSTM, MODEL_FASTTEXT_SIMPLE, \
    MODEL_ML, MODEL_FASTTEXT_DEEP_ANN, DIR_IMAGES_EDA, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_REPORT
from training.training import ModelTrainer


def main():
    for dir in [DIR_REPORT, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_IMAGES_EDA]:
        make_dir_if_not_exists(dir)
    name = 'incorrect_pred_removed'
    # get cleaned train and test data
    preprocessor = PreProcessor()
    data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()

    # exploratory data analysis
    try:
        eda = EDA(data, name)
        eda.visualize()
        eda.analyze()
    except Exception as e:
        print(e)

    # Train and evaluation
    trainer = ModelTrainer(name, data, data_test)
    trainer.train_tfidf_ml(MODEL_ML)

    try:
        run_automl_test()
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()