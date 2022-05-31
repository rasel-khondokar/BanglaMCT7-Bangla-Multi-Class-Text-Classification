from exploratory_analysis.eda import EDA
from preprocessing.preprocessing import PreProcessor
from scraping.helpers import make_dir_if_not_exists
from settings import DIR_REPORT, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_IMAGES_EDA

for dir in [DIR_REPORT, DIR_IMAGES_HISTORY, DIR_PERFORMENCE_REPORT, DIR_IMAGES_EDA]:
    make_dir_if_not_exists(dir)
name = 'incorrect_pred_removed'
# get cleaned train and test data
preprocessor = PreProcessor()
data, data_test = preprocessor.read_collected_data_incorrect_pred_removed()

# # exploratory data analysis
eda = EDA(data, name)
eda.visualize()