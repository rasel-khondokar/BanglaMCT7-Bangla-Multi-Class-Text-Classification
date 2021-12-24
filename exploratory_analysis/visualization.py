import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scraping.helpers import make_dir_if_not_exists
from settings import DIR_IMAGES_EDA

class DataVisualizer():

    def __init__(self, data):
        self.data = data

    def show_class_distribution(self):
        sns.set(font_scale=1.4)
        self.data['category'].value_counts().plot(kind='barh', figsize=(8, 6))
        plt.xlabel("Number of Articles", labelpad=12)
        plt.ylabel("Category", labelpad=12)
        plt.yticks(rotation = 45)
        plt.title("Dataset Distribution", y=1.02)
        file_path = DIR_IMAGES_EDA + '/class_distribution.png'
        plt.savefig(file_path)
        plt.close()
        print(f'class distribution image saved to - {file_path}')

    def show_document_length_distribution(self):
        self.data['Length'] = self.data.cleaned.apply(lambda x: len(x.split()))
        matplotlib.rc_file_defaults()
        frequency = dict()
        for i in self.data.Length:
            frequency[i] = frequency.get(i, 0) + 1

        plt.figure(figsize=(6, 4))
        plt.bar(frequency.keys(), frequency.values(), color=(0.2, 0.4, 0.6, 0.6))
        plt.xlim(21, 700)

        plt.xlabel('Length of the Documents')
        plt.ylabel('Frequency')
        plt.title('Length-Frequency Distribution')
        file_path = DIR_IMAGES_EDA + '/document_length_distribution.png'
        plt.savefig(file_path)
        plt.close()
        print(f'document length distribution image saved to - {file_path}')

        print(f"Maximum Length of a Document: {max(self.data.Length)}")
        print(f"Minimum Length of a Document: {min(self.data.Length)}")
        print(f"Average Length of a Document: {round(np.mean(self.data.Length), 0)}")

    def show_data_summary(self):
        print('\n______________________showing data summary ________________________________\n')
        documents = []
        words = []
        u_words = []
        # find class names
        class_label = [k for k, v in self.data.category.value_counts().to_dict().items()]
        for label in class_label:
            word_list = [word.strip().lower() for t in list(self.data[self.data.category == label].cleaned) for word in
                         t.strip().split()]
            counts = dict()
            for word in word_list:
                counts[word] = counts.get(word, 0) + 1
            # sort the dictionary of word list
            ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            # Documents per class
            documents.append(len(list(self.data[self.data.category == label].cleaned)))
            # Total Word per class
            words.append(len(word_list))
            # Unique words per class
            u_words.append(len(np.unique(word_list)))

            print("\nClass Name : ", label)
            print("Number of Documents:{}".format(len(list(self.data[self.data.category == label].cleaned))))
            print("Number of Words:{}".format(len(word_list)))
            print("Number of Unique Words:{}".format(len(np.unique(word_list))))
            print("Most Frequent Words:\n")
            for k, v in ordered[:10]:
                print("{}\t{}".format(k, v))

        data_matrix = pd.DataFrame({'Total Documents': documents,
                                    'Total Words': words,
                                    'Unique Words': u_words,
                                    'Class Names': class_label})
        print('\n______________________summary________________________________\n')
        print(data_matrix)

        df = pd.melt(data_matrix, id_vars="Class Names", var_name="Category", value_name="Values")
        plt.figure(figsize=(8, 6))
        ax = plt.subplot()

        sns.barplot(data=df, x='Class Names', y='Values', hue='Category')
        ax.set_xlabel('Class Names')
        ax.set_title('Data Statistics')
        class_names = class_label
        ax.xaxis.set_ticklabels(class_names, rotation=45)
        file_path = DIR_IMAGES_EDA + '/data_summary.png'
        plt.savefig(file_path)
        plt.close()
