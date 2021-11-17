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

    def scrape_from_url(self, driver, url):
        print(f'Retrieving {url}')
        driver.get(url)
        time.sleep(PUSH_TIME_FOR_REQUEST_TO_URL)
        return driver


    def get_elelment_by_text(self, element, text):
        element = element.find_element_by_xpath(f"//*[contains(text(), '{text}')]")
        return element

    def get_data_from_texts_parent_element(self, driver, text):
        element = self.get_elelment_by_text(driver, text)
        element = element.find_element_by_xpath("..")
        element_text = element.text
        element_text = element_text.replace(text, '')
        return element_text.strip()

    def add_data_from_company_detail(self, data):

        company =  data['company_name'].replace(' ', '%20')
        license_number =  data['license_number']

        url = f'https://govservices.dcra.dc.gov/contractorratingsystem/ProfessionalProjects?prof={company}&type=General-Contractor&cnumber={license_number}'

        driver = self.driver
        time.sleep(DELAY_LONG)
        # open new tab
        driver.execute_script(f"window.open('{url}', 'new_window')")
        # Switch to the tab
        driver.switch_to.window(driver.window_handles[1])
        time.sleep(DELAY_LONG)
        try:
            # Wait untill page is loaded
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.row:nth-child(5) span')))
            try:
                data['business_address]'] = self.get_data_from_texts_parent_element(driver, 'Business Address:')
            except Exception as e:
                print(e)
            try:
                data['business_phone'] = self.get_data_from_texts_parent_element(driver, 'Business Phone:')
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)

        # Back to the main window
        time.sleep(DELAY_LONG)
        driver.switch_to_window(driver.window_handles[0])
        time.sleep(DELAY_LONG)

        return data


    def get_posts(self, driver):

        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#actorProjectScores .modal-content')))
        posts = driver.find_elements_by_css_selector('#actorProjectScores .modal-content')

        post_data = []
        for post in posts:
            time.sleep(DELAY_LONG)
            data = {}
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h3.text-uppercase')))
            company = post.find_element_by_css_selector('h3.text-uppercase')
            data['company_name'] = company.text.strip()
            if 'fire' in data['company_name'].lower() or 'safety' in data['company_name'].lower():
                continue

            data['license_number'] = post.find_element_by_css_selector('.lblprofessional:nth-child(3) span').text.strip().replace('License Number: ', '')
            data['stop_work_order'] = post.find_element_by_css_selector('.lblprofessional:nth-child(7)').text.strip().replace('Stop Work Order(s): ', '')
            data['project_count'] = post.find_element_by_css_selector('br~ .lblprofessional+ .lblprofessional span').text.strip().replace('Project Count: ', '')
            data['business_email'] = post.find_element_by_css_selector('.lblprofessional:nth-child(5) span').text.strip().replace('Business Email: ', '')
            self.add_data_from_company_detail(data)
            post_data.append(data)

        return post_data

    def get_from_next_page(self, i, driver, last_page):
        posts = self.get_posts(driver)
        if i != last_page:
            time.sleep(20)
            next_page = driver.find_element_by_xpath(f"//div[@class='Pager']/a[@page='{i + 1}']")
            next_page.click()
        return posts

    def get_posts_posts_page(self, driver, urls, existing, category, data_file):
        for url in urls:
            # print(url.get_attribute('innerHTML'))
            if len(url.text) != 0:
                link = url.get_attribute('href')

                if link:

                    if url not in existing:

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

                        add_to_existing_json(data_dict, data_file)

                        # Back to the main window
                        time.sleep(2)
                        driver.switch_to.window(driver.window_handles[0])
                        time.sleep(2)


    def scroll_to_element(self, driver, el: WebElement):
        driver.execute_script("arguments[0].scrollIntoView(true);", el)
        time.sleep(3)

    def scrape(self):
        data_file =  f'{BASE_DIR}/DATASET/prothomalo.json'
        driver = self.driver
        main_site = 'https://www.prothomalo.com/'
        categories = {'sports':'sports', 'international':'world', 'economy':'business', 'entertainment':'entertainment',
                      'technology':'education/science-tech', 'politics':'politics'}

        try:
            with open(data_file, "r") as the_file:
                existing = json.load(the_file)
            existing = [data['url'] for data in existing]
        except:
            existing = []

        SCRAPING_STATUS = True
        for category in categories:

            driver.get(main_site + categories[category])
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.stories-set')))

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

                    self.get_posts_posts_page(driver, urls, existing, category, data_file)

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

def main_prothom_alo():
    # chromedriver_autoinstaller.install(True)
    time.sleep(10)
    chrome_version = chromedriver_autoinstaller.get_chrome_version()
    driver_dcra = get_driver('https://www.prothomalo.com/', chrome_version = chrome_version, headless=False)
    scraper = DcraScraper(driver_dcra)
    scraper.scrape()
    driver_dcra.quit()

# if __name__ == "__main__":
#     main_prothom_alo()