import os
import pandas as pd
from configs import BASE_DIR

# categories = {'sports': 'sports', 'international': 'international', 'economy': 'economy',
#               'entertainment': 'entertainment',
#               'technology': 'technology', 'politics': 'politics', 'education': 'education'}


total = 0
total_dup = 0
dataset_dir = f'{BASE_DIR}/DATASET/'
files = os.listdir(dataset_dir)
for file in files:
    if '.json' in file:
        print(f"{file}")
        try:
            data_file =  dataset_dir+file
            df = pd.read_json(data_file)
            total_cat = len(df)
            print(f"Total len :{total_cat}")
            total+=total_cat
            total_cat_dup = df.duplicated('url').sum()
            print(f"Duplicated :{total_cat_dup}")
            total_dup+total_cat_dup
        except:
            print("error")

print(f"Total valid :{total-total_dup}")


dataset = pd.DataFrame()
dataset_dir = f'{BASE_DIR}/DATASET/'
files = os.listdir(dataset_dir)
for file in files:
    if '.json' in file:
        try:
            data_file = dataset_dir + file
            df = pd.read_json(data_file)
            # time.sleep(10)
            dataset = dataset.append(df, ignore_index=True)
        except:
            print(f"error during reading {file}")

dataset.rename(columns={"raw_text": "cleanText"}, inplace=True)
file = f'{dataset_dir}collected_data.csv'
dataset.to_csv(file, index=False)

df =  pd.read_csv(file)
print(len(df))

