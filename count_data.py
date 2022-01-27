import os
import pandas as pd
from configs import BASE_DIR

# categories = {'sports': 'sports', 'international': 'international', 'economy': 'economy',
#               'entertainment': 'entertainment',
#               'technology': 'technology', 'politics': 'politics', 'education': 'education'}

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
        except Exception as e:
            print(f"error during reading {file}")
            print(e)

dataset.rename(columns={"raw_text": "cleanText"}, inplace=True)
file = f'{dataset_dir}collected_data.csv'
dataset.to_csv(file, index=False)
df =  pd.read_csv(file)
total = len(df)
duplicated = df.duplicated('url').sum()
print(f"Total :{total}")
print(f"Valid :{total-duplicated}")
print(f"Duplicated :{duplicated}")
print(df['category'].value_counts())

