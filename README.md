# BanglaMCT7 : Bangla Multi-Class Text Classification
#### [Click to download the dataset](https://www.kaggle.com/gakowsher/banglamct7-bangla-multiclass-text-dataset-7-tags/code)
#### [Click to download trained models](https://drive.google.com/file/d/16BgSgZO1JMRCPo7_uZXxXmPaXwDmiVoc/view?usp=sharing)

## Installation and training

#### Installing, Downloading Prerequisites and Traning

```sh
. install_and_train.sh
```
#### Only Traning

```sh
python __main__.py
```

## Preprocessing

**Train data**

Before removing null : 98883 

After removing null : 98883

Before removing duplicates : 98883

After removing duplicates : 98426

1893 Documents removed which length less than equal 20

**Test data**

Before removing null : 98884

After removing null : 98884

Before removing duplicates : 98884

After removing duplicates : 98386



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
| BIDIRECTIONAL GRU  | keras tokenaizer  | 0.98  | 0.93|
| CNN & BIDIRECTIONAL LSTM  | keras tokenaizer  | 0.95  |0.93|
| DEEP ANN  | fasttest tokenaizer  | 0.96  | 0.92|
| SIMPLE ANN  | fasttest tokenaizer  | 0.94  |0.93|
| RANDOM FOREST  | tf-idf tokenaizer  | 0.97  |0.91|

**By considering overfitting and stable performance of pre-trained model I am selecting SIMPLE ANN (2 dense layers) with fasttext tokenaizer
model as the best model.**


<!-- TABLE_GENERATE_END -->

## Best Model

#### SIMPLE ANN (2 dense layers) with fasttext tokenaizer

#### Model file path :  _`resources/simple_fasttext_best_model.pkl`_

#### Model summary

![summary](REPORT/IMAGES/best_model_summary.png?raw=true)

#### Model history

![accuracy](REPORT/IMAGES/HISTORY/simple_fasttext_accuracy.png?raw=true)
![loss](REPORT/IMAGES/HISTORY/simple_fasttext_loss.png?raw=true)

**_Note : The callback function will save the model if accuracy is improved during the epoch._**

#### Report - train data

    ___________________ confusion_matrix _____________________
    [[12795   115    61   133    23   895   390]
     [   67 11655    74    11    20   513   136]
     [   17    36 14526   122    57   175    40]
     [  133    22   250 14532    37   233   331]
     [   10     3    71    44 15507    26     8]
     [  439   301   237   152    26 14117    88]
     [  294    86    50   165    14   105  9284]]


    ___________________ classification report _____________________
                  precision    recall  f1-score   support
    
               0       0.93      0.89      0.91     14412
               1       0.95      0.93      0.94     12476
               2       0.95      0.97      0.96     14973
               3       0.96      0.94      0.95     15538
               4       0.99      0.99      0.99     15669
               5       0.88      0.92      0.90     15360
               6       0.90      0.93      0.92      9998
    
        accuracy                           0.94     98426
       macro avg       0.94      0.94      0.94     98426
    weighted avg       0.94      0.94      0.94     98426

#### Report - test data

    ___________________ confusion_matrix _____________________
    [[12680   163    67   160    28   918   451]
     [   82 11563   112    23    18   634   183]
     [   21    62 14380   168    79   227    70]
     [  165    39   269 14081    65   304   381]
     [    7    19    99    63 15633    42    18]
     [  485   463   267   160    34 13668   104]
     [  386   112    61   232    19   118  9003]]


    ___________________ classification report _____________________
                  precision    recall  f1-score   support
    
               0       0.92      0.88      0.90     14467
               1       0.93      0.92      0.92     12615
               2       0.94      0.96      0.95     15007
               3       0.95      0.92      0.93     15304
               4       0.98      0.98      0.98     15881
               5       0.86      0.90      0.88     15181
               6       0.88      0.91      0.89      9931
    
        accuracy                           0.93     98386
       macro avg       0.92      0.92      0.92     98386
    weighted avg       0.93      0.93      0.93     98386

## Flask Web APP

**_app.py_**

![ui](REPORT/IMAGES/app.png?raw=true)
