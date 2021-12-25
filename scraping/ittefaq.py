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
            print(url.get_attribute('innerHTML'))
            link = url.get_attribute('href')

            if link:

                if link not in existing:

                    data_dict = {'title': url.get_attribute('title'), 'category': category}

                    # open new tab
                    driver.execute_script(f"window.open('{link}', 'new_window')")
                    # Switch to the tab
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(2)

                    try:
                        data_dict['url'] = driver.current_url
                    except Exception as e:
                        print(e)
                        data_dict['url'] = link

                    try:
                        data_dict['published_date'] = driver.find_element_by_css_selector('.tts_time').text
                    except Exception as e:
                        print(e)

                    try:
                        content_detail = driver.find_elements_by_css_selector('.content_detail_content_outer')[0]
                        paragraphs = content_detail.find_elements_by_css_selector('p.alignfull')
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

    def scrape(self, categories, category, main_site):
        data_file =  f'{BASE_DIR}/DATASET/ittefaq_{category}.json'
        driver = self.driver

        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        print(len(existing))

        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.link_overlay')))

        page_count = 1
        SCRAPING_STATUS = True
        while SCRAPING_STATUS:
            try:
                print(f'Page no : {page_count}')
                driver.get(main_site + f'{categories[category]}{page_count}')
                urls = driver.find_elements_by_css_selector('a.link_overlay')
                try:
                    self.get_posts_posts_page(driver, urls, existing, category, data_file)
                except:
                    pass
                page_count+=1
            except Exception as e:
                print(e)
                SCRAPING_STATUS = False
                break

def main_ittefaq(categories, category):
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    main_site = 'https://www.ittefaq.com.bd/'
    driver_dcra = get_driver(main_site, chrome_version = chrome_version, headless=True)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape(categories, category, main_site)
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()