import json
import os
import time

from get_chrome_driver import GetChromeDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Request to URL using Chrome driver
from configs import CHROMEDRIVER_PATH, BASE_DIR


def get_driver(url, headless=True, chrome_version=False):
    option = Options()
    option.add_argument("--disable-notifications")
    option.add_argument('--no-sandbox')
    option.add_argument('--disable-dev-shm-usage')

    if headless:
        option.add_argument("--headless")

    chrome_dir = BASE_DIR+'/'+CHROMEDRIVER_PATH
    make_dir_if_not_exists(chrome_dir)
    chrome_file_path = chrome_dir + '/chromedriver'

    try:
        driver = webdriver.Chrome(chrome_file_path, chrome_options=option)
        driver.get(url)
    except Exception as e:
        print('Selenium session is not Created !')

        if os.path.exists(chrome_file_path):
            os.remove(chrome_file_path)
            print(f'Removed {chrome_file_path} file!')

        download_driver = GetChromeDriver()
        download_driver.auto_download(extract=True, output_path=chrome_dir)
        print(f'Downloaded chrome driver for the chrome version {download_driver.matching_version()}!')
        driver = webdriver.Chrome(chrome_file_path, chrome_options=option)
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
        json.dump(existing, the_file, indent=4, ensure_ascii=False)

    print(f'Saved to {file}')
