import threading
import time
from datetime import datetime

from scraping.prothom_alo import main_prothom_alo


class ScraperThread (threading.Thread):
   def __init__(self,categories, name):
      threading.Thread.__init__(self)
      self.threadID = name
      self.name = name
      self.categories = categories

   def run(self):
      print ("Starting " + self.name)
      print(datetime.now())
      main_prothom_alo(self.categories, self.name)
      print(datetime.now())
      print ("Exiting " + self.name)


def scrape_prothom_alo():
   categories = {'sports':'sports', 'international':'world', 'economy':'business', 'entertainment':'entertainment',
                         'technology':'education/science-tech', 'politics':'politics', 'education':'education'}

   # thread = ScraperThread(categories, 'sports')
   # thread.start()

   for category in categories:
      time.sleep(3)
      thread = ScraperThread(categories, category)
      thread.start()
