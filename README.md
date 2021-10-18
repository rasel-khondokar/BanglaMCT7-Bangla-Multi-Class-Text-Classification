# BanglaMCT7 : Bangla-Multi Class Text Classification
#### [Click to download the dataset](https://www.kaggle.com/gakowsher/banglamct7-bangla-multiclass-text-dataset-7-tags/code)

## Installation and training

#### Installing, Downloading Prerequisites and Traning

```sh
. install_and_train.sh
```
#### Only Traning

```sh
python __main__.py
```

## EDA

EDA images saved to :  _`REPORT/IMAGES/EDA`_

![class distribution](REPORT/IMAGES/EDA/class_distribution.png?raw=true)
![summary](REPORT/IMAGES/EDA/data_summary.png?raw=true)
![document length distribution](REPORT/IMAGES/EDA/document_length_distribution.png?raw=true)

## Trained models accuracy

Trained models accuracy & loss history images saved to :  _`REPORT/IMAGES/HISTORY`_

Trained models performence report saved to :  _`REPORT/PERFORMENCE`_

<!-- TABLE_GENERATE_START -->

| Model  | Tokenaizer | Train | Test |
| --- | ---- |---- |---- |
| BIDIRECTIONAL GRU  | keras tokenaizer  | 0.99  | 0.99|
| CNN & BIDIRECTIONAL LSTM  | keras tokenaizer  | 0.99  |0.99|
| DEEP ANN  | fasttest tokenaizer  | 0.99  | 0.99|
| SIMPLE ANN  | fasttest tokenaizer  | 0.99  |0.99|
| RANDOM FOREST  | tf-idf tokenaizer  | 0.99  |0.99|

**By considering overfitting and stable performance of pre-trained model I am selecting fasttext bidirectional LSTM
model as the best model.**

<!-- TABLE_GENERATE_END -->

## Best Model

#### fasttext bi lstm model

#### Model history

![accuracy](REPORT/IMAGES/HISTORY/bi_gru_keras_tokenaizer_accuracy.png?raw=true)
![loss](REPORT/IMAGES/HISTORY/bi_gru_keras_tokenaizer_loss.png?raw=true)

#### Classification Report - train data

              precision    recall  f1-score   support

           0       0.91      0.88      0.90     14467
           1       0.95      0.90      0.93     12615
           2       0.95      0.93      0.94     15007
           3       0.93      0.92      0.93     15304
           4       0.97      0.98      0.97     15881
           5       0.91      0.88      0.89     15181
           6       0.79      0.94      0.86      9931

      accuracy                         0.92     98386
      macro avg    0.92      0.92      0.92     98386
      weighted avg 0.92      0.92      0.92     98386

#### Classification Report - test data

              precision    recall  f1-score   support

           0       0.91      0.88      0.90     14467
           1       0.95      0.90      0.93     12615
           2       0.95      0.93      0.94     15007
           3       0.93      0.92      0.93     15304
           4       0.97      0.98      0.97     15881
           5       0.91      0.88      0.89     15181
           6       0.79      0.94      0.86      9931

      accuracy                         0.92     98386
      macro avg    0.92      0.92      0.92     98386
      weighted avg 0.92      0.92      0.92     98386
