import pandas as pd
from configs import BASE_DIR

categories = {'sports': 'sports', 'international': 'world', 'economy': 'business', 'entertainment': 'entertainment',
              'technology': 'education/science-tech', 'politics': 'politics', 'education': 'education'}


for category in categories:
    data_file =  f'{BASE_DIR}/DATASET/prothomalo_{category}.json'
    df = pd.read_json(data_file)
    print(f"{category}")
    print(f"Total len :{len(df)}")
    print(f"Duplicated :{df.duplicated('url').sum()}")

