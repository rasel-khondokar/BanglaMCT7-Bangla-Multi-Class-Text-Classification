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
from scraping.helpers import get_driver, add_to_existing_json


class DcraScraper():
    def __init__(self, driver):
        self.driver = driver

    def get_posts_posts_page(self, driver, urls, existing, category, data_file):
        for url in urls:
            # print(url.get_attribute('innerHTML'))
            if len(url.text) != 0:
                link = url.get_attribute('href')

                if link:

                    if link not in existing:

                        data_dict = {'url': link, 'title': url.text, 'category': category}

                        # open new tab
                        driver.execute_script(f"window.open('{link}', 'new_window')")
                        # Switch to the tab
                        time.sleep(2)
                        driver.switch_to.window(driver.window_handles[1])
                        time.sleep(2)

                        # WebDriverWait(driver, 20).until(
                        #     EC.presence_of_element_located((By.CSS_SELECTOR, '.businessCard--businessName')))

                        try:
                            data_dict['published_date'] = driver.find_element(By.CSS_SELECTOR,
                                                                              '.storyPageMetaData-m__publish-time__19bdV').text
                        except Exception as e:
                            print(e)
                        try:
                            paragraphs = driver.find_elements_by_css_selector('p')
                            text = ''
                            for paragraph in paragraphs:
                                text += f'{paragraph.text}\n\n'
                            data_dict['raw_text'] = text
                        except Exception as e:
                            print(e)

                        try:
                            add_to_existing_json(data_dict, data_file)
                        except:
                            pass

                        # Back to the main window
                        time.sleep(2)
                        driver.switch_to.window(driver.window_handles[0])
                        time.sleep(2)


    def scroll_to_element(self, driver, el: WebElement):
        driver.execute_script("arguments[0].scrollIntoView(true);", el)
        time.sleep(3)

    def scrape(self, categories, category):
        data_file =  f'{BASE_DIR}/DATASET/prothomalo_{category}.json'
        driver = self.driver
        main_site = 'https://www.prothomalo.com/'
        # categories = {'sports':'sports', 'international':'world', 'economy':'business', 'entertainment':'entertainment',
        #               'technology':'education/science-tech', 'politics':'politics'}
        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        print(len(existing))

        driver.get(main_site + categories[category])
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.stories-set')))

        SCRAPING_STATUS = True
        while SCRAPING_STATUS:
            try:

                headings = driver.find_elements_by_css_selector('.bn-story-card h2')
                urls = []
                for heading in headings:
                    urls.append( heading.find_element_by_xpath('..') )

                # Try to set last date as first date to check only new jobs
                try:
                    first_index = last_date_index
                except:
                    first_index = 0
                last_date_index = len(urls) - 1

                urls = urls[first_index:last_date_index]

                try:
                    self.get_posts_posts_page(driver, urls, existing, category, data_file)
                except:
                    pass

                # load more
                try:
                    load_more = driver.find_element_by_css_selector('.load-more-content')
                    self.scroll_to_element(driver, load_more)

                    javascript = "document.querySelector('.load-more-content').click();"
                    driver.execute_script(javascript)
                    time.sleep(10)
                except Exception as e:
                    print(e)
                print(0)
            except Exception as e:
                print(e)
                SCRAPING_STATUS = False
                break

def main_prothom_alo(categories, category):
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    driver_dcra = get_driver('https://www.prothomalo.com/', chrome_version = chrome_version, headless=True)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape(categories, category)
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()