import json
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Request to URL using Chrome driver
def get_driver(url, headless=True, chrome_version=False):

    option = Options()
    option.add_argument("--disable-notifications")
    option.add_argument('--no-sandbox')
    option.add_argument('--disable-dev-shm-usage')

    if headless:
        option.add_argument("--headless")

    if chrome_version:
        print(f'Chrome version : {chrome_version}')
        # driver_path = chrome_version.split('.')[0]
        driver_path = 'chromedriver_linux64'
        dir_path = os.path.dirname(os.path.realpath(__file__))+'/'+driver_path+'/chromedriver'
        driver = webdriver.Chrome(dir_path, chrome_options=option)
    else:
        driver = webdriver.Chrome(chrome_options=option)

    driver.get(url)
    driver.maximize_window()
    return driver

def make_dir_if_not_exists(file_path):
    dirs = file_path.split('/')
    if dirs:
        path = ''
        for dir in dirs:
            if dir:
                path = path + dir + '/'
                if not os.path.exists(path):
                    os.mkdir(path)

def add_to_existing_json(data, file):
    dirs = file.split('/')

    try:
        if len(dirs)>=2:
            dir = dirs[:-1]
            make_dir_if_not_exists('/'.join(dir))
    except Exception as e:
        print(e)

    try:
        with open(file, "r") as the_file:
            existing = json.load(the_file)
    except FileNotFoundError as e:
        existing = []
    existing.append(data)

    with open(file, "w") as the_file:
        json.dump(existing, the_file, indent=4)

    print(f'Saved to {file}')
