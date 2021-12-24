import threading
import time
from datetime import datetime

from scraping.banglatribune import main_banglatribune
from scraping.jagonews24 import main_jagonews24
from scraping.prothom_alo import main_prothom_alo


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


