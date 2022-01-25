						# https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit
										# Домашнее задание № 4 (демо).
							# Обнаружение сарказма с помощью логистической регрессии.

# Мы будем использовать набор данных из статьи «Большой самоаннотированный корпус для сарказма» 
# с > 1 млн комментариев от Reddit, помеченных как саркастические, так и нет. Обработанную версию 
# можно найти на Kaggle в виде набора данных Kaggle.

# Обнаружить сарказм легко.
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter('ignore')

train_df = pd.read_csv('../sarcasm/train-balanced-sarcasm.csv')

# print(train_df.head())
# print(train_df.info())

# Некоторые комментарии отсутствуют, поэтому мы отбрасываем соответствующие строки.

train_df.dropna(subset=['comment'], inplace=True)

# Мы замечаем, что набор данных действительно сбалансирован
# print(train_df.info())

# print(train_df['label'].value_counts())

# Мы разделяем данные на обучающую и проверочную (validation) части.

train_texts, valid_texts, y_train, y_valid = train_test_split(train_df['comment'], train_df['label'], random_state=17)

# 1) Проанализируйте набор данных, сделайте несколько графиков. Это ядро может служить примером:
# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

# Часть 1. Исследовательский анализ данных.
# Распределение длин саркастических и нормальных комментариев практически одинаково.
# train_df.loc[train_df['label'] == 1, 'comment'].str.len().apply(np.log1p).hist(label='sarcastic', alpha=.5)
# train_df.loc[train_df['label'] == 0, 'comment'].str.len().apply(np.log1p).hist(label='normal', alpha=.5)
# plt.legend();

# from wordcloud import WordCloud, STOPWORDS
# wordcloud = WordCloud(background_color='black', stopwords = STOPWORDS,
#                 max_words = 200, max_font_size = 100, 
#                 random_state = 17, width=800, height=400)
# Облако слов (wordcloud) - это хорошо, но не очень полезно

# plt.figure(figsize=(10, 7))
# wordcloud.generate(str(train_df.loc[train_df['label'] == 1, 'comment']))
# plt.imshow(wordcloud);

# plt.figure(figsize=(11, 8))
# wordcloud.generate(str(train_df.loc[train_df['label'] == 0, 'comment']))
# plt.imshow(wordcloud);

# Давайте проанализируем, являются ли одни субреддиты в среднем более «саркастичными», чем другие.
# sub_df = train_df.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
# print(sub_df.sort_values(by='sum', ascending=False).head(10))

# print(sub_df[sub_df['size'] > 1000].sort_values(by='mean', ascending=False).head(10))

# То же самое для авторов не дает особого понимания. За исключением того, 
# что были отобраны чьи-то комментарии - мы можем видеть одинаковое количество 
# саркастических и несаркастических комментариев.
# sub_df = train_df.groupby('author')['label'].agg([np.size, np.mean, np.sum])
# print(sub_df[sub_df['size'] > 300].sort_values(by='mean', ascending=False).head(10))

# sub_df = train_df[train_df['score'] >= 0].groupby('score')['label'].agg([np.size, np.mean, np.sum])
# print(sub_df[sub_df['size'] > 300].sort_values(by='mean', ascending=False).head(10))

# sub_df = train_df[train_df['score'] < 0].groupby('score')['label'].agg([np.size, np.mean, np.sum])
# print(sub_df[sub_df['size'] > 300].sort_values(by='mean', ascending=False).head(10))


							# Часть 2. Обучение модели.

# 2) Создайте конвейер логистической регрессии Tf-Idf + для прогнозирования сарказма (метки, label) 
# на основе текста комментария на Reddit (комментарий, comment).

# построить биграммы, установить ограничение на максимальное количество функций и минимальную частоту слов

tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
# полиномиальная логистическая регрессия, также известная как классификатор softmax
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', random_state=17, verbose=1)
# sklearn's pipeline
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), ('logit', logit)])

tfidf_logit_pipeline.fit(train_texts, y_train)

valid_pred = tfidf_logit_pipeline.predict(valid_texts)

print(accuracy_score(y_valid, valid_pred))

							# Часть 3. Объяснение модели.

# 3) Нанесите на график слова / биграммы, наиболее предсказывающие сарказм (для этого 
# можно использовать eli5)

# def plot_confusion_matrix(actual, predicted, classes,
#                           normalize=False,
#                           title='Confusion matrix', figsize=(7,7),
#                           cmap=plt.cm.Blues, path_to_save_fig=None):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     import itertools
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(actual, predicted).T
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#     plt.figure(figsize=figsize)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('Predicted label')
#     plt.xlabel('True label')
    
#     if path_to_save_fig:
#         plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')

# Матрица путаницы достаточно сбалансирована.

# plot_confusion_matrix(y_valid, valid_pred, tfidf_logit_pipeline.named_steps['logit'].classes_, figsize=(8, 8))

# Действительно, мы можем распознать некоторые фразы, указывающие на сарказм. Типа «да, конечно».

# import eli5																														???
# print(eli5.show_weights(estimator=tfidf_logit_pipeline.named_steps['logit'],vec=tfidf_logit_pipeline.named_steps['tf_idf']))

								# Часть 4. Улучшение модели

# 4) (необязательно) добавить субреддиты в качестве новых функций для повышения производительности 
# модели. Примените здесь подход Bag of Words, то есть рассматривайте каждый субреддит как новую функцию.

subreddits = train_df['subreddit']
train_subreddits, valid_subreddits = train_test_split(subreddits, random_state=17)

# У нас будут отдельные векторизаторы Tf-Idf для комментариев и для сабреддитов. Можно также придерживаться конвейера,
# но в этом случае это становится немного менее простым. Пример

tf_idf_texts = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
tf_idf_subreddits = TfidfVectorizer(ngram_range=(1, 1))

# Выполняйте преобразования отдельно для комментариев и субреддитов.

X_train_texts = tf_idf_texts.fit_transform(train_texts)
X_valid_texts = tf_idf_texts.transform(valid_texts)

print(X_train_texts.shape, X_valid_texts.shape)

X_train_subreddits = tf_idf_subreddits.fit_transform(train_subreddits)
X_valid_subreddits = tf_idf_subreddits.transform(valid_subreddits)

print(X_train_subreddits.shape, X_valid_subreddits.shape)

# Затем сложите все функции вместе.

from scipy.sparse import hstack
X_train = hstack([X_train_texts, X_train_subreddits])
X_valid = hstack([X_valid_texts, X_valid_subreddits])

print(X_train.shape, X_valid.shape)

# Обучите ту же логистическую регрессию.

logit.fit(X_train, y_train)

valid_pred = logit.predict(X_valid)

print(accuracy_score(y_valid, valid_pred))

# Как видим, точность немного увеличилась.





plt.show()












