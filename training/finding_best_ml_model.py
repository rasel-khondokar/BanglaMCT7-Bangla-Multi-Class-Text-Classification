from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from preprocessing.preprocessing import PreProcessor


def get_best_model():
    name = 'auto_ml'

    preprocessor = PreProcessor()
    data = preprocessor.read_collected_data_incorrect_pred_removed(is_split=False)

    corpus = preprocessor.vectorize_tfidf(data.cleaned, (1, 1), name)
    labels, class_names = preprocessor.encode_category(data.category)

    X_train, X_valid, y_train, y_valid = train_test_split(corpus, labels,
                                                          test_size=0.2, random_state=0)

    # Fit into the models
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, config_dict="TPOT sparse" )
    tpot.fit(X_train, y_train)
    print(tpot.score(X_valid, y_valid))
    tpot.export('tpot_best_pipeline.py')

if __name__=='__main__':
    get_best_model()
