import os
from pathlib import Path

CHROMEDRIVER_PATH = 'chromedriver_linux64'
PUSH_TIME_FOR_REQUEST_TO_URL = 3
MAIN_SITE = 'https://govservices.dcra.dc.gov/contractorratingsystem/BuildingProfessionals/BuildingProfessional?profType=General%20Contractor&profName='
DELAY_LONG = 5
DELAY_SHORT = 3
OUTPUT_FILENAME = 'output.xlsx'

ROOT_DIR = Path(__file__).parent.parent
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CHROMEDRIVER_PATH = 'chromedriver_linux64'