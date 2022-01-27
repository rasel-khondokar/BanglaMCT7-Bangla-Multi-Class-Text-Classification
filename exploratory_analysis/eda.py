from exploratory_analysis.analysis import DataAnalyzer
from exploratory_analysis.visualization import DataVisualizer

class EDA:

    def __init__(self, data, name):
        self.data = data
        self.name = name

    def visualize(self):
        visualizer = DataVisualizer(self.data, self.name)
        visualizer.show_class_distribution()
        visualizer.show_document_length_distribution()
        visualizer.show_data_summary()

    def analyze(self):
        analyzer = DataAnalyzer(self.data)
        analyzer.show_cleaned_data()