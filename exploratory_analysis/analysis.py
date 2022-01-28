
class DataAnalyzer():

    def __init__(self, data):
        self.data = data

    def show_cleaned_data(self):
        sample_data = self.data.sample(10)
        sample_data = sample_data.reset_index()
        print('\n______________________showing cleaned data________________________________\n')
        for row in range(len(sample_data)):
            print('Original:\n', sample_data.loc[row]['cleaned'], '\nCleaned:\n',
                  sample_data.loc[row]['cleaned'], '\n', 'category:-- ', sample_data.loc[row]['category'], '\n')



