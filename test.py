import pandas as pd
from configs import BASE_DIR

categories = {'sports': 'sport/news?tags=', 'international': 'foreign/news?page=', 'economy': '/business-all?page=',
              'entertainment': 'entertainment/news?page=',
              'technology': 'tech-and-gadget/news?page=', 'politics': 'politics?page=', 'education': 'educations?page='}

for category in categories:
    try:
        data_file =  f'{BASE_DIR}/DATASET/banglatribune_{category}.json'
        df = pd.read_json(data_file)
        print(f"{category}")
        print(f"Total len :{len(df)}")
        print(f"Duplicated :{df.duplicated('url').sum()}")
    except:
        print("error")
