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

Before removing null : 95134 

After removing null : 85800

Before removing duplicates : 85800

After removing duplicates : 70028

837 Documents removed which length less than equal 20

Total Remaining Documents: 69191

**Test data**

Before removing null : 23784

After removing null : 21523

Before removing duplicates : 21523

After removing duplicates : 17806



## EDA

Maximum Length of a Document: 4442 

Average Length of a Document: 223


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
| BIDIRECTIONAL GRU  | keras tokenaizer  | 0.98  | 0.94|
| CNN & BIDIRECTIONAL LSTM  | keras tokenaizer  | 0.98  |0.94|
| DEEP ANN  | fasttest tokenaizer  | 0.97  | 0.95|
| SIMPLE ANN  | fasttest tokenaizer  | 0.96  |0.95|
| RANDOM FOREST  | tf-idf tokenaizer  | 0.93  |0.91|
| bert-base-multilingual-cased  | bert-base-multilingual-cased  | 1.00  |0.91|
| csebuetnlp/banglabert  | csebuetnlp/banglabert  | 0.99  |0.93|
| monsoon-nlp/bangla-electra  | monsoon-nlp/bangla-electra  | 1.00  |0.94|
| sagorsarker/bangla-bert-base  | sagorsarker/bangla-bert-base  | 1.00  |0.93|

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
    [[ 9708    71    19   173    67    12   234]
     [   96  7616    20    88    81     3   294]
     [   36    23 10220    79    49    23    25]
     [  130    34    84  9213    32    22   154]
     [   62    59    20    31 10862     8     9]
     [   22     8    57    69    12 14356    10]
     [  259    96    42   298    14    10  5118]]
    
    
    ___________________ classification report _____________________
                   precision    recall  f1-score   support
    
          economy       0.94      0.94      0.94     10284
        education       0.96      0.93      0.95      8198
    entertainment       0.98      0.98      0.98     10455
    international       0.93      0.95      0.94      9669
         politics       0.98      0.98      0.98     11051
           sports       0.99      0.99      0.99     14534
       technology       0.88      0.88      0.88      5837
    
         accuracy                           0.96     70028
        macro avg       0.95      0.95      0.95     70028
     weighted avg       0.96      0.96      0.96     70028
#### Report - test data
    
    ___________________ confusion_matrix _____________________
    [[2582   21    5   40   27    4   70]
     [  24 1898    6   24   25    3  144]
     [  10    9 2444   31   21   11   16]
     [  43   13   28 2307    7    6   48]
     [  16   11    8    5 2768    6    5]
     [   7    2   18   12    8 3629    2]
     [  71   50   12   80    8    2 1219]]
    
    
    ___________________ classification report _____________________
                   precision    recall  f1-score   support
    
          economy       0.94      0.94      0.94      2749
        education       0.95      0.89      0.92      2124
    entertainment       0.97      0.96      0.97      2542
    international       0.92      0.94      0.93      2452
         politics       0.97      0.98      0.97      2819
           sports       0.99      0.99      0.99      3678
       technology       0.81      0.85      0.83      1442
    
         accuracy                           0.95     17806
        macro avg       0.94      0.94      0.94     17806
     weighted avg       0.95      0.95      0.95     17806

## Flask Web APP

**_app.py_**

![ui](REPORT/IMAGES/app.png?raw=true)
