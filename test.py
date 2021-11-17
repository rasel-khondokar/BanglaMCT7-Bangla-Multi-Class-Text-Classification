import pandas as pd

from settings import DIR_DATASET

data = pd.read_csv(DIR_DATASET + '/BanglaMCT7/train.csv')
state = data[data["category"] == 'state']
print(state)