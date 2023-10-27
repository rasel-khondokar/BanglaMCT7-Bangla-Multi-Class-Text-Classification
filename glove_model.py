import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from preprocessing.preprocessing import PreProcessor


# def change_matplotlib_font(font_download_url):
#     FONT_PATH = 'MY_FONT'
#
#     font_download_cmd = f"wget {font_download_url} -O {FONT_PATH}.zip"
#     unzip_cmd = f"unzip -o {FONT_PATH}.zip -d {FONT_PATH}"
#     os.system(font_download_cmd)
#     os.system(unzip_cmd)
#
#     font_files = fm.findSystemFonts(fontpaths=FONT_PATH)
#     for font_file in font_files:
#         fm.fontManager.addfont(font_file)
#
#     font_name = fm.FontProperties(fname=font_files[0]).get_name()
#     matplotlib.rc('font', family=font_name)
#
#
# font_download_url = "https://fonts.google.com/download?family=Noto+Serif+Bengali"
# change_matplotlib_font(font_download_url)
preprocessor = PreProcessor()
data = preprocessor.read_collected_data_incorrect_pred_removed(is_split=False)
cleaned = data['cleaned'].to_list()
sentences = []
for i in cleaned:
    sentences.append(i.split())
# print(sentences)

# train model
model = Word2Vec(sentences, min_count=1, size=300)
model.save('model_glove_300.bin')
model.wv.save_word2vec_format('model_glove_word2vec_format_300.bin')
model.wv.save_word2vec_format('model_glove_word2vec_format_300.txt', binary=False)

# new_model = Word2Vec.load('model_glove_100.bin')
# # fit a 2d PCA model to the vectors
# X = new_model[new_model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# plt.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# plt.savefig('glove.png')