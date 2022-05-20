import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scraping.helpers import make_dir_if_not_exists
from settings import DIR_IMAGES_EDA

class DataVisualizer():

    def __init__(self, data, name):
        self.data = data
        self.name = name

    def show_class_distribution(self):
        class_distribution = self.data['category'].value_counts(sort=True, ascending=False)
        class_distribution = class_distribution.to_frame()
        class_distribution = class_distribution.reset_index()
        fig, ax = plt.subplots()
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        ax.barh(class_distribution['index'], class_distribution['category'], color="green")
        for i, v in enumerate(class_distribution['category']):
            ax.text(v, i, str(v), in_layout=True)
        plt.xlabel("Number of Articles")
        plt.ylabel("Category")
        plt.yticks(rotation = 45)
        plt.title("Dataset Distribution")
        file_path = DIR_IMAGES_EDA + f'/{self.name}_class_distribution.eps'
        plt.savefig(file_path, format='eps', bbox_inches='tight')
        plt.close()
        print(f'class distribution image saved to - {file_path}')

    def show_document_length_distribution(self):
        self.data['Length'] = self.data.cleaned.apply(lambda x: len(x.split()))
        matplotlib.rc_file_defaults()
        frequency = dict()
        fig, ax = plt.subplots()
        right, left, top = ax.spines["right"], ax.spines["left"], ax.spines["top"]
        right.set_visible(False)
        left.set_visible(False)
        top.set_visible(False)
        for i in self.data.Length:
            frequency[i] = frequency.get(i, 0) + 1
        plt.bar(frequency.keys(), frequency.values(), color=(0.2, 0.4, 0.6, 0.6))
        plt.xlabel('Length of the Documents')
        plt.ylabel('Frequency')
        plt.title('Length-Frequency Distribution')
        file_path = DIR_IMAGES_EDA + f'/{self.name}_document_length_distribution.eps'
        plt.savefig(file_path, format='eps', bbox_inches='tight')
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
        right, left, top = ax.spines["right"], ax.spines["left"], ax.spines["top"]
        right.set_visible(False)
        left.set_visible(False)
        top.set_visible(False)
        g = sns.barplot(data=df, x='Class Names', y='Values', hue='Category')

        ax2 = g
        # annotate axis = seaborn axis
        for p in ax.patches:
            ax2.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width(), p.get_height()),
                        ha='center', va='center', xytext=(0, 2),
                        textcoords='offset points')

        ax.set_xlabel('Class Names')
        ax.set_title('Data Statistics')
        class_names = class_label
        ax.xaxis.set_ticklabels(class_names, rotation=45)
        file_path = DIR_IMAGES_EDA + f'/{self.name}_data_summary.eps'
        plt.savefig(file_path, format='eps', bbox_inches='tight')
        plt.close()
