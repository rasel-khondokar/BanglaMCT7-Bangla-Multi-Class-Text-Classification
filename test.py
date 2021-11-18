import pandas as pd
#
# from settings import DIR_DATASET
#
# data = pd.read_csv(DIR_DATASET + '/BanglaMCT7/train.csv')
# state = data[data["category"] == 'state']
# print(state)
from configs import BASE_DIR

data_file =  f'{BASE_DIR}/DATASET/prothomalo.json'
df = pd.read_json(data_file)
print(df.duplicated('url').sum())

