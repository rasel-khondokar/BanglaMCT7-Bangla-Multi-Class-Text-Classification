import json
import time
import pandas as pd
import selenium
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller

from configs import ROOT_DIR, BASE_DIR
from scraping.helpers import get_driver, add_to_existing_json, make_dir_if_not_exists


class DcraScraper():
    def __init__(self, driver):
        self.driver = driver

    def get_posts_posts_page(self, driver, urls, existing, category, data_file):
        for url in urls:
            # print(url.get_attribute('innerHTML'))
            # print(url.get_attribute('href'))
            try:
                link = url.get_attribute('href')

                if link:

                    if link not in existing:
                        existing.append(link)

                        data_dict = {'url': link, 'category': category}

                        # open new tab
                        driver.execute_script(f"window.open('{link}', 'new_window')")
                        # Switch to the tab
                        time.sleep(2)
                        driver.switch_to.window(driver.window_handles[1])
                        time.sleep(5)

                        # WebDriverWait(driver, 20).until(
                        #     EC.presence_of_element_located((By.CSS_SELECTOR, '.businessCard--businessName')))

                        try:
                            data_dict['title'] = driver.find_element_by_css_selector('.post-heading h1').text
                        except Exception as e:
                            print(e)

                        try:
                            data_dict['published_date'] = driver.find_element_by_css_selector('.post-heading .time').text
                        except Exception as e:
                            print(e)

                        try:
                            paragraphs = driver.find_elements_by_css_selector('.news-article p')
                            text = ''
                            for paragraph in paragraphs[:-1]:
                                text += f'{paragraph.text}\n\n'
                            data_dict['raw_text'] = text
                        except Exception as e:
                            print(e)

                        try:
                            add_to_existing_json(data_dict, data_file)
                        except Exception as e:
                            print(e)

                        # Back to the main window
                        time.sleep(2)
                        driver.switch_to.window(driver.window_handles[0])
                        time.sleep(2)
            except Exception as e:
                print(e)

    def scroll_to_element(self, driver, el: WebElement):
        driver.execute_script("arguments[0].scrollIntoView(true);", el)
        time.sleep(3)

    def scrape(self, categories, category):
        data_dir = f'DATASET'
        make_dir_if_not_exists(data_dir)
        # data_file =  f'{BASE_DIR}/{data_dir}/banglanews24_{category}.json' # for current year
        data_file =  f'{BASE_DIR}/{data_dir}/banglanews24_{category}_{categories[category][-5:-1]}.json' # for recent years
        driver = self.driver
        main_site = 'https://www.banglanews24.com/'
        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        print(len(existing))

        cat_page = main_site + categories[category]
        driver.get(cat_page)
        # WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.loadMoreButton')))

        # Try to set last date as first date to check only new jobs
        try:
            pg_no = driver.find_elements_by_css_selector('.page-link')
            max = int(pg_no[-2].text)

            for page in range(1, max + 1):
                # driver.get(f'{cat_page}?page={page}') # for current year
                driver.get(f'{cat_page}page={page}') # for previous years

                try:
                    urls = driver.find_elements_by_css_selector('.category-area .list a' )
                except Exception as e:
                    print(e)

                try:
                    self.get_posts_posts_page(driver, urls, existing, category, data_file)
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

def main_banglanews24(categories, category):
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    driver_dcra = get_driver('https://www.banglanews24.com/', chrome_version = chrome_version, headless=True)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape(categories, category)
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()