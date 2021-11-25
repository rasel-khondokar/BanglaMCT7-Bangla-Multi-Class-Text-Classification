import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from exploratory_analysis.eda import EDA
from preprocessing.preprocessing import PreProcessor
from settings import MODEL_SELECTED, MODEL_BIDIRECTIONAL_GRU, MODEL_CNN_BIDIRECTIONAL_LSTM, MODEL_FASTTEXT_SIMPLE, \
    MODEL_ML, MODEL_FASTTEXT_DEEP_ANN
from training.training import ModelTrainer


def main():

    # get cleaned train and test data
    preprocessor = PreProcessor()
    data, data_test = preprocessor.read_collected_data()

    # exploratory data analysis
    eda = EDA(data)
    eda.visualize()
    eda.analyze()

    # Train and evaluation
    trainer = ModelTrainer(data, data_test)
    trainer.train_keras_tokenaizer(MODEL_BIDIRECTIONAL_GRU)
    trainer.train_keras_tokenaizer(MODEL_CNN_BIDIRECTIONAL_LSTM)
    trainer.train_fasttext(MODEL_FASTTEXT_SIMPLE)
    trainer.train_fasttext(MODEL_FASTTEXT_DEEP_ANN)
    trainer.train_tfidf_ml(MODEL_ML)


if __name__=='__main__':
    main()