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
                        time.sleep(2)

                        # WebDriverWait(driver, 20).until(
                        #     EC.presence_of_element_located((By.CSS_SELECTOR, '.businessCard--businessName')))

                        try:
                            data_dict['title'] = driver.find_element_by_css_selector('h1.fontSize32').text
                        except Exception as e:
                            print(e)

                        try:
                            data_dict['published_date'] = driver.find_element_by_css_selector('.detail-time').text
                        except Exception as e:
                            print(e)

                        try:
                            content =  driver.find_elements_by_css_selector('.contentBody.appBodyPaddingLR .description')[0]
                            paragraphs = content.find_elements_by_css_selector('p')
                            text = ''
                            for paragraph in paragraphs:
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
        data_file =  f'{BASE_DIR}/{data_dir}/samakal_{category}.json'
        driver = self.driver
        main_site = 'https://www.samakal.com/'
        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        print(len(existing))

        driver.get(main_site + categories[category])
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.loadMoreButton')))

        SCRAPING_STATUS = True
        while SCRAPING_STATUS:
            try:

                urls = driver.find_elements_by_css_selector('.child-cat-list .link-overlay')

                # Try to set last date as first date to check only new jobs
                try:
                    first_index = last_date_index
                except:
                    first_index = 0
                last_date_index = len(urls) - 1

                urls = urls[first_index:last_date_index]

                try:
                    # print(0)
                    self.get_posts_posts_page(driver, urls, existing, category, data_file)
                except:
                    pass

                # load more
                try:
                    loadMoreButton = '.loadMoreButton'
                    load_more = driver.find_element_by_css_selector(f'{loadMoreButton}')
                    self.scroll_to_element(driver, load_more)

                    javascript = f"document.querySelector('{loadMoreButton}').click();"
                    driver.execute_script(javascript)
                    time.sleep(5)
                except Exception as e:
                    print(e)

                    print(f"Cant load more uring scraping {category}")

            except Exception as e:
                print(e)
                SCRAPING_STATUS = False
                break

def main_samakal(categories, category):
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    driver_dcra = get_driver('https://samakal.com/', chrome_version = chrome_version, headless=True)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape(categories, category)
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()