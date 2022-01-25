					### https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-4-linear-classification-and-regression-44a41b9b5220
					#   Оpen Machine Learning Course. 
					# 	Topic 4. Linear Classification and Regression


# Добро пожаловать на 4-ю неделю нашего курса! Теперь мы представим нашу самую 
# важную тему - линейные модели. Если у вас есть подготовленные данные и вы 
# хотите приступить к обучению моделей, то, скорее всего, вы сначала попробуете 
# либо линейную, либо логистическую регрессию, в зависимости от вашей задачи 
# (регрессия или классификация).

# Материал этой недели охватывает как теорию линейных моделей, так и практические 
# аспекты их использования в реальных задачах. В этой теме будет много математики, 
# и мы даже не будем пытаться отображать все формулы на Medium. Вместо этого мы 
# предоставляем блокнот Jupyter для каждой части этой статьи. В этом задании вы 
# пройдете два простых теста в конкурсе Kaggle, решив задачу идентификации 
# пользователя на основе его сеанса посещенных веб-сайтов.


							# Часть 1. Регрессия
# 		https://nbviewer.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic04_linear_models/topic4_linear_models_part1_mse_likelihood_bias_variance.ipynb?flush_cache=true#1.-Introduction

# Ordinary Least Squares - Обычные наименьшие квадраты
# Maximum Likelihood Estimation - Оценка максимального правдоподобия
# Bias-Variance Decomposition - Разложение дисперсии смещения
# Regularization of Linear Regression - Регуляризация линейной регрессии
# Конспект - в тетради

						# Часть 2. Логистическая регрессия
#		http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic04_linear_models/topic4_linear_models_part2_logit_likelihood_learning.ipynb?flush_cache=true

# Линейный классификатор
# Логистическая регрессия как линейный классификатор
# Принцип максимального правдоподобия и логистическая регрессия
# L2-регуляризация логистической функции потерь
# Конспект - в тетради
		
			#Часть 3. Наглядный пример регуляризации логистической регрессии

# В первой статье мы продемонстрировали, как полиномиальные функции позволяют линейным моделям 
# строить нелинейные разделяющие поверхности. Покажем теперь это наглядно.

# Давайте посмотрим, как регуляризация влияет на качество классификации набора данных при тестировании 
# микрочипов из курса Эндрю Нг по машинному обучению. Мы будем использовать логистическую регрессию с 
# полиномиальными функциями и варьировать параметр регуляризации C. Во-первых, мы увидим, как регуляризация 
# влияет на разделяющую границу классификатора, и интуитивно распознаем недостаточное и избыточное соответствие. 
# Затем мы выберем параметр регуляризации, численно близкий к оптимальному значению с помощью (cross-validation) 
# и (GridSearch).

# нам не нравятся предупреждения
# вы можете прокомментировать следующие 2 строки, если хотите
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.model_selection import (GridSearchCV, StratifiedKFold,cross_val_score)
# from sklearn.preprocessing import PolynomialFeatures

# Загрузим данные с помощью read_csv из библиотеки pandas. В этом наборе данных по 118 микрочипам 
# (объектам) есть результаты двух тестов контроля качества (две числовые переменные) и информация о том, 
# был ли микрочип запущен в производство. Переменные уже центрированы, что означает, что из значений 
# столбца были вычтены собственные средние значения. Таким образом, «средний» микрочип соответствует 
# нулевому значению в результатах тестирования.

# Загрузка данных
# data = pd.read_csv("../mlcourse.ai/data/microchip_tests.txt", header=None, names=("test1", "test2", "released"))
# получение информации о фрейме данных
# print(data.info())
# Давайте посмотрим на первую и последнюю 5 строк.
# print(data.head(5))
# print(data.tail(5))

# Теперь мы должны сохранить обучающий набор и метки целевого класса в отдельных массивах NumPy.
# X = data.iloc[:, :2].values
# y = data.iloc[:, 2].values

# В качестве промежуточного шага мы можем построить данные. 
# Оранжевые точки соответствуют дефектным чипам, синие - нормальным.

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
# plt.xlabel("Test 1")
# plt.ylabel("Test 2")
# plt.title("2 tests of microchips. Logit with C=1")
# plt.legend();

# Давайте определим функцию для отображения разделительной кривой классификатора. 

# def plot_boundary(clf, X, y, grid_step=0.01, poly_featurizer=None):
#     x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
#     y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))

#     # до каждой точки от [x_min, m_max] x [y_min, y_max]
#     # ставим в соответствие свой цвет
#     Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
#     Z = Z.reshape(xx.shape)
#     plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# Создадим объект sklearn, который добавит в матрицу X полиномиальные признаки вплоть 
# до степени 7 и обучим логистическую регрессию с параметром регуляризации C = 10^{-2}. 
# Изобразим разделяющую границу.
# poly = PolynomialFeatures(degree=7)
# X_poly = poly.fit_transform(X)
# print(X_poly.shape)

# C = 1e-2
# logit = LogisticRegression(C=C, random_state=17)
# logit.fit(X_poly, y)

# plot_boundary(logit, X, y, grid_step=0.01, poly_featurizer=poly)

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
# plt.xlabel("Test 1")
# plt.ylabel("Test 2")
# plt.title("2 tests of microchips. Logit with C=%s" % C)
# plt.legend()

# Также проверим долю правильных ответов классификатора на обучающей выборке. Видим, что 
# регуляризация оказалась слишком сильной, и модель "недообучилась". Доля правильных ответов 
# классификатора на обучающей выборке оказалась равной 0.627.

# print("Accuracy on training set:", round(logit.score(X_poly, y), 3))

# Увеличим C до 1. Тем самым мы ослабляем регуляризацию, теперь в решении значения весов 
# логистической регрессии могут оказаться больше (по модулю), чем в прошлом случае. Теперь 
# доля правильных ответов классификатора на обучающей выборке – 0.831.

# C = 1
# logit = LogisticRegression(C=C, random_state=17)
# logit.fit(X_poly, y)

# plot_boundary(logit, X, y, grid_step=0.005, poly_featurizer=poly)

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
# plt.xlabel("Test 1")
# plt.ylabel("Test 2")
# plt.title("2 tests of microchips. Logit with C=%s" % C)
# plt.legend()

# print("Accuracy on training set:", round(logit.score(X_poly, y), 3))

# Еще увеличим C – до 10 тысяч. Теперь регуляризации явно недостаточно, и мы наблюдаем 
# переобучение. Можно заметить, что в прошлом случае (при C=1 и "гладкой" границе) доля 
# правильных ответов модели на обучающей выборке не намного ниже, чем в 3 случае, зато на 
# новой выборке, можно себе представить, 2 модель сработает намного лучше.
# Доля правильных ответов классификатора на обучающей выборке – 0.873.

# C = 1e4
# logit = LogisticRegression(C=C, random_state=17)
# logit.fit(X_poly, y)

# plot_boundary(logit, X, y, grid_step=0.005, poly_featurizer=poly)

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
# plt.xlabel("Test 1")
# plt.ylabel("Test 2")
# plt.title("2 tests of microchips. Logit with C=%s" % C)
# plt.legend()

# print("Accuracy on training set:", round(logit.score(X_poly, y), 3))

					# Настройка параметра регуляризации

# Теперь найдем оптимальное (в данном примере) значение параметра регуляризации C. Сделать 
# это можно с помощью LogisticRegressionCV – перебора параметров по сетке с последующей 
# кросс-валидацией. Этот класс создан специально для логистической регрессии (для нее известны 
# эффективные алгоритмы перебора параметров), для произвольной модели мы бы использовали GridSearchCV, 
# RandomizedSearchCV или, например, специальные алгоритмы оптимизации гиперпараметров, реализованные 
# в hyperopt.

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

# c_values = np.logspace(-2, 3, 500)

# logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=-1)
# logit_searcher.fit(X_poly, y)

# print(logit_searcher.C_)

# Посмотрим, как качество модели (доля правильных ответов на обучающей и валидационной выборках) 
# меняется при изменении гиперпараметра C.
# plt.plot(c_values, np.mean(logit_searcher.scores_[1], axis=0))
# plt.xlabel("C")
# plt.ylabel("Mean CV-accuracy");

# Наконец, выберите область с «лучшими» значениями C.

# plt.plot(c_values, np.mean(logit_searcher.scores_[1], axis=0))
# plt.xlabel("C")
# plt.ylabel("Mean CV-accuracy")
# plt.xlim((0, 10));

# Как мы помним, такие кривые называются валидационными, раньше мы их строили вручную, но в sklearn 
# для них их построения есть специальные методы, которые мы тоже сейчас будем использовать.
# plt.show()

			

						# Тема 4. Линейная классификация и регрессия.
				# Часть 4. Где логистическая регрессия хороша, а где нет
# https://nbviewer.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic04_linear_models/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.ipynb

# 1) Анализ обзоров фильмов IMDB
# 2) Простой подсчет слов
# 3) XOR-проблема

					# 1. Анализ обзоров фильмов IMDB

# А теперь немного практики! Мы хотим решить проблему бинарной классификации обзоров фильмов IMDB. 
# У нас есть обучающий набор с отмеченными отзывами, 12500 отзывов отмечены как хорошие, еще 12500 - плохие. 
# Здесь непросто сразу начать заниматься машинным обучением, потому что у нас нет матрицы X нам нужно его 
# подготовить. Мы будем использовать простой подход: мешка слов ("Bag of words"). Особенности обзора будут представлены 
# индикаторами наличия каждого слова из всего корпуса в этом обзоре. Корпус - это совокупность всех отзывов 
# пользователей. Идея иллюстрируется картинкой

from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Для начала мы автоматически загружаем набор данных и разархивируем его вместе с остальными наборами 
# данных в папке данных. Набор данных кратко описан здесь. В тестовой и обучающей выборках 12,5 тыс. 
# хороших и плохих отзывов.

# reviews_train = load_files("../aclImdb/train", categories=["pos", "neg"])
# text_train, y_train = reviews_train.data, reviews_train.target
# print("Number of documents in training data: %d" % len(text_train))
# print(np.bincount(y_train))
# # поменяйте путь к файлу
# reviews_test = load_files("../aclImdb/test", categories=["pos", "neg"])
# text_test, y_test = reviews_test.data, reviews_test.target
# print("Number of documents in test data: %d" % len(text_test))
# print(np.bincount(y_test))

# Вот несколько примеров отзывов.

# print(text_train[1])
# print(y_train[1])	 # bad review

# print(text_train[2])
# print(y_train[2]) 	 # good review

# 2. Простой подсчет слов
# Составим словарь всех слов с помощью CountVectorizer. Всего в выборке 74849 уникальных слов. 
# Если посмотреть на примеры полученных "слов" (лучше их называть токенами), то можно увидеть, 
# что многие важные этапы обработки текста мы тут пропустили (автоматическая обработка текстов – 
# это могло бы быть темой отдельной серии статей).

# cv = CountVectorizer()
# cv.fit(text_train)

# print(len(cv.vocabulary_))
# print(cv.get_feature_names()[:50])
# print(cv.get_feature_names()[50000:50050])

# Закодируем предложения из текстов обучающей выборки индексами входящих слов. 
# Используем разреженный формат. Преобразуем так же тестовую выборку.

# X_train = cv.transform(text_train)
# X_test = cv.transform(text_test)

# Посмотрим, как работала наша трансформация
# print(text_train[19726])
# print(X_train[19726].nonzero()[1])
# print(X_train[19726].nonzero())


# Обучим логистическую регрессию и посмотрим на доли правильных ответов на обучающей и тестовой 
# выборках. Получается, на тестовой выборке мы правильно угадываем тональность примерно 86.7% 
# отзывов.

# logit = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7)
# logit.fit(X_train, y_train)
# print(round(logit.score(X_train, y_train), 3), round(logit.score(X_test, y_test), 3))

# Коэффициенты модели можно красиво отобразить.
# def visualize_coefficients(classifier, feature_names, n_top_features=25):
#     # get coefficients with large absolute values
#     coef = classifier.coef_.ravel()
#     positive_coefficients = np.argsort(coef)[-n_top_features:]
#     negative_coefficients = np.argsort(coef)[:n_top_features]
#     interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
#     # plot them
#     plt.figure(figsize=(15, 5))
#     colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
#     plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
#     feature_names = np.array(feature_names)
#     plt.xticks(
#         np.arange(1, 1 + 2 * n_top_features),
#         feature_names[interesting_coefficients],
#         rotation=60,
#         ha="right",
#     );

# def plot_grid_scores(grid, param_name):
#     plt.plot(
#         grid.param_grid[param_name],
#         grid.cv_results_["mean_train_score"],
#         color="green",
#         label="train",
#     )
#     plt.plot(
#         grid.param_grid[param_name],
#         grid.cv_results_["mean_test_score"],
#         color="red",
#         label="test",
#     )
#     plt.legend();


# visualize_coefficients(logit, cv.get_feature_names())


# Подберем коэффициент регуляризации для логистической регрессии. Используем sklearn.pipeline, 
# поскольку CountVectorizer правильно применять только на тех данных, на которых в текущий момент 
# обучается модель (чтоб не "подсматривать" в тестовую выборку и не считать по ней частоты 
# вхождения слов). В данном случае pipeline задает последовательность действий: применить 
# CountVectorizer, затем обучить логистическую регрессию. Так мы поднимаем долю правильных 
# ответов до 88.5% на кросс-валидации и 87.9% – на отложенной выборке.

from sklearn.pipeline import make_pipeline

# text_pipe_logit = make_pipeline(CountVectorizer(), 
# LogisticRegression(n_jobs=-1, random_state=7))

# text_pipe_logit.fit(text_train, y_train)
# print(text_pipe_logit.score(text_test, y_test))

from sklearn.model_selection import GridSearchCV

# param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
# grid_logit = GridSearchCV(text_pipe_logit, param_grid_logit, cv=3, n_jobs=-1)

# grid_logit.fit(text_train, y_train)

# Напечатаем лучший C и cv-score с использованием этого гиперпараметра:
# grid_logit.best_params_, grid_logit.best_score_
# plot_grid_scores(grid_logit, 'logisticregression__C')
# print(grid_logit.score(text_test, y_test))																	# ???

# Теперь то же самое, но со случайным лесом. Видим, что с логистической регрессией 
# мы достигаем большей доли правильных ответов меньшими усилиями. Лес работает дольше, 
# на отложенной выборке 85.5% правильных ответов.

from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=17)
# forest.fit(X_train, y_train)
# print(round(forest.score(X_test, y_test), 3))

	
									# 3. XOR-проблема

# Теперь рассмотрим пример, где линейные модели справляются хуже.

# Линейные методы классификации строят все же очень простую разделяющую поверхность – 
# гиперплоскость. Самый известный игрушечный пример, в котором классы нельзя без ошибок 
# поделить гиперплоскостью (то есть прямой, если это 2D), получил имя "the XOR problem".

# XOR – это "исключающее ИЛИ", булева функция со следующей таблицей истинности:

# 		0 		1
# 0 	0 		1
# 1 	1 		0

# XOR дал имя простой задаче бинарной классификации, в которой классы представлены 
# вытянутыми по диагоналям и пересекающимися облаками точек.

# creating dataset
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired);

# Очевидно, что невозможно без ошибок провести одну прямую линию, чтобы отделить один класс от другого. 
# Таким образом, логистическая регрессия плохо справляется с этой задачей.

def plot_boundary(clf, X, y, plot_title):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    clf.fit(X, y)
    # построить функцию принятия решения для каждой точки данных в сетке
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)

    image = plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes="--")
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    plt.title(plot_title, fontsize=12);

plot_boundary(LogisticRegression(solver="lbfgs"), X, y, "Logistic Regression, XOR problem")

# Но если на входе ввести полиномиальные характеристики (здесь до 2 степеней), то проблема решена.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

logit_pipe = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2)),
        ("logit", LogisticRegression(solver="lbfgs")),
    ]
)



plot_boundary(logit_pipe, X, y, "Logistic Regression + quadratic features. XOR problem")

# Здесь логистическая регрессия все равно строила гиперплоскость, но в 6-мерном пространстве 
# признаков $1, x_1, x_2, x_1^2, x_1x_2$ и $x_2^2$. В проекции на исходное пространство 
# признаков $x_1, x_2$ граница получилась нелинейной.

# На практике полиномиальные признаки действительно помогают, но строить их явно – вычислительно 
# неэффективно. Гораздо быстрее работает SVM с ядровым трюком. При таком подходе в пространстве 
# высокой размерности считается только расстояние между объектами (задаваемое функцией-ядром), 
# а явно плодить комбинаторно большое число признаков не приходится. Про это подробно можно 
# почитать в курсе Евгения Соколова (математика уже серьезная).





plt.show()

# 6. Плюсы и минусы линейных моделей в задачах машинного обучения

# Плюсы:

# Хорошо изучены
# Очень быстрые, могут работать на очень больших выборках
# Практически вне конкуренции, когда признаков очень много (от сотен тысяч и более), 
# и они разреженные (хотя есть еще факторизационные машины)
# Коэффициенты перед признаками могут интерпретироваться (при условии что признаки масштабированы) 
# – в линейной регрессии как частные производные зависимой переменной от признаков, в логистической
# – как изменение шансов на отнесение к одному из классов в $\exp^{\beta_i}$ раз при изменении 
# признака $x_i$ на 1 ед., подробнее тут
# Логистическая регрессия выдает вероятности отнесения к разным классам (это очень ценится, 
# например, в кредитном скоринге)
# Модель может строить и нелинейную границу, если на вход подать полиномиальные признаки

# Минусы:
# Плохо работают в задачах, в которых зависимость ответов от признаков сложная, нелинейная
# На практике предположения теоремы Маркова-Гаусса почти никогда не выполняются, поэтому чаще 
# линейные методы работают хуже, чем, например, SVM и ансамбли (по качеству решения задачи 
# классификации/регрессии)












