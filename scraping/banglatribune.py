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
            # print(url.get_attribute('href'))

            link = url.get_attribute('href')

            if link:

                if link not in existing:

                    data_dict = {'url': link, 'category': category}

                    try:
                        data_dict['title'] = url.get_attribute('title')
                    except Exception as e:
                        print(e)

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
                                                                          '.tts_time').text
                    except Exception as e:
                        print(e)
                    try:
                        paragraphs = driver.find_elements_by_css_selector('.content_item_number0 .alignfull , .content_item_number0 strong')
                        text = ''
                        for paragraph in paragraphs[:-1]:
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
                else:
                    print('Already exists!')


    def scroll_to_element(self, driver, el: WebElement):
        driver.execute_script("arguments[0].scrollIntoView(true);", el)
        time.sleep(3)

    def scrape(self, categories, category):
        data_file =  f'{BASE_DIR}/DATASET/banglatribune_{category}.json'
        driver = self.driver
        main_site = 'https://www.banglatribune.com/'

        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        print(len(existing))


        SCRAPING_STATUS = True
        page_no = 1
        driver.get(main_site + categories[category])
        while SCRAPING_STATUS:
            print(f'Page no. : {page_no}')
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.contents_listing.widget')))


                urls = driver.find_elements_by_css_selector('.col3 .link_overlay')

                try:
                    self.get_posts_posts_page(driver, urls, existing, category, data_file)
                except:
                    pass

                # load more
                try:
                    load_more = driver.find_element_by_css_selector('.next_page')
                    self.scroll_to_element(driver, load_more)

                    javascript = "document.querySelector('.next_page').click();"
                    driver.execute_script(javascript)
                    time.sleep(3)
                except Exception as e:
                    print(e)

            except Exception as e:
                print(e)
                SCRAPING_STATUS = False
                break

            page_no+=1

def main_banglatribune(categories, category):
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    driver_dcra = get_driver('https://www.banglatribune.com/', chrome_version = chrome_version, headless=True)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape(categories, category)
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()