from exploratory_analysis.analysis import DataAnalyzer
from exploratory_analysis.visualization import DataVisualizer

class EDA:

    def __init__(self, data):
        self.data = data

    def visualize(self):
        visualizer = DataVisualizer(self.data)
        visualizer.show_class_distribution()
        visualizer.show_document_length_distribution()
        visualizer.show_data_summary()

    def analyze(self):
        analyzer = DataAnalyzer(self.data)
        analyzer.show_cleaned_data()