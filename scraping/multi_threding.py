import threading
import time
from datetime import datetime

from scraping.banglanews24 import main_banglanews24
from scraping.banglatribune import main_banglatribune
from scraping.ittefaq import main_ittefaq
from scraping.jagonews24 import main_jagonews24
from scraping.prothom_alo import main_prothom_alo
from scraping.samakal import main_samakal


class ScraperThread (threading.Thread):
   def __init__(self,categories, name, scraper):
      threading.Thread.__init__(self)
      self.threadID = name
      self.name = name
      self.categories = categories
      self.scraper = scraper

   def run(self):
      print ("Starting " + self.name)
      print(datetime.now())
      self.scraper(self.categories, self.name)
      print(datetime.now())
      print ("Exiting " + self.name)


def scrape_prothom_alo():
   categories = {'sports':'sports', 'international':'world', 'economy':'business', 'entertainment':'entertainment',
                         'technology':'education/science-tech', 'politics':'politics', 'education':'education'}

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category)
      thread.start()

def scrape_banglatribune():
   categories = {'sports':'sport/news?tags=', 'international':'foreign/news?page=', 'economy':'/business-all?page=', 'entertainment':'entertainment/news?page=',
                         'technology':'tech-and-gadget/news?page=', 'politics':'politics?page=', 'education':'educations?page='}

   # categories = {'entertainment':'entertainment/news?page='}

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category, main_banglatribune)
      thread.start()

def scrape_jagonews24():
   categories = {'sports':'sports', 'international':'international', 'economy':'economy', 'entertainment':'entertainment',
                         'technology':'technology', 'politics':'politics', 'education':'education'}

   categories = {'sports':'sports'}

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category, main_jagonews24)
      thread.start()


def scrape_ittefaqe():
   categories = {'sports':'sports', 'international':'world-news', 'economy':'business', 'entertainment':'entertainment',
                         'technology':'tech', 'politics':'politics', 'education':'education'}

   # categories = {'sports':'sports'}
   for cat in categories:
      categories[cat] = cat + '?page='

   for category in categories:
      time.sleep(5)
      thread = ScraperThread(categories, category, main_ittefaq)

def scrape_samakal():
   categories = {'international':'international', 'economy':'economics',
                         'technology':'technology', 'education':'education'}

   # categories = {'technology':'technology'}

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category, main_samakal)
      thread.start()


def scrape_banglanews24():
   categories = {'international':'category/আন্তর্জাতিক/4', 'economy':'category/অর্থনীতি-ব্যবসা/3',
                         'technology':'category/তথ্যপ্রযুক্তি/7', 'education':'category/শিক্ষা/20'}

   # categories = {'international':'category/আন্তর্জাতিক/4'}

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category, main_banglanews24)
      thread.start()

def scrape_banglanews24_from_recent_years():

   years = [2019, 2018, 2017, 2016, 2015, 2014]
   # years = [2019]
   for year in years:
      years_q = f'category/international/4?y={year}&'

      categories = {'international':years_q}

      for category in categories:
         time.sleep(3)
         thread = ScraperThread(categories, category, main_banglanews24)
         thread.start()


