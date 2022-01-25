					### https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-4-linear-classification-and-regression-44a41b9b5220
					#   Оpen Machine Learning Course. 
					# 	Topic 4. Linear Classification and Regression


					# Часть 5. Кривые валидации и обучения
# from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
# import warnings
# warnings.filterwarnings('ignore')
# import seaborn as sns


# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV, SGDClassifier)
# from sklearn.model_selection import learning_curve, validation_curve
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Мы уже получили представление о проверке модели, кросс-валидации и регуляризации.
# Теперь рассмотрим главный вопрос:

# Если качество модели нас не устраивает, что делать?

# Сделать модель сложнее или упростить?
# Добавить больше признаков?
# Или нам просто нужно больше данных для обучения?

# Ответы на данные вопросы не всегда лежат на поверхности. В частности, иногда 
# использование более сложной модели приведет к ухудшению показателей. Либо 
# добавление наблюдений не приведет к ощутимым изменениям. Способность принять 
# правильное решение и выбрать правильный способ улучшения модели, собственно 
# говоря, и отличает хорошего специалиста от плохого.

# Будем работать со знакомыми данными по оттоку клиентов телеком-оператора.

# data = pd.read_csv('../mlcourse.ai/data/telecom_churn.csv').drop('State', axis=1)
# data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
# data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})

# y = data['Churn'].astype('int').values
# X = data.drop('Churn', axis=1).values

# Логистическую регрессию будем обучать стохастическим градиентным спуском. Пока 
# объясним это тем, что так быстрее, но далее в программе у нас отдельная статья 
# про это дело. 

# alphas = np.logspace(-2, 0, 20)
# sgd_logit = SGDClassifier(loss="log", n_jobs=-1, random_state=17, max_iter=5)
# logit_pipe = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("poly", PolynomialFeatures(degree=2)),
#         ("sgd_logit", sgd_logit),
#     ]
# )
# val_train, val_test = validation_curve(logit_pipe, X, y, param_name = "sgd_logit__alpha",param_range = alphas, cv=5, scoring="roc_auc")

# Построим валидационные кривые, показывающие, как качество (ROC AUC) на 
# обучающей и проверочной выборке меняется с изменением параметра регуляризации.

# def plot_with_err(x, data, **kwargs):
#     mu, std = data.mean(1), data.std(1)
#     lines = plt.plot(x, mu, "-", **kwargs)
#     plt.fill_between(
#         x,
#         mu - std,
#         mu + std,
#         edgecolor="none",
#         facecolor=lines[0].get_color(),
#         alpha=0.2,
#     )

# plot_with_err(alphas, val_train, label="training scores")
# plot_with_err(alphas, val_test, label="validation scores")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("ROC AUC")
# plt.legend()
# plt.grid(True);

# Тенденция видна сразу, и она очень часто встречается.

# Для простых моделей тренировочная и валидационная ошибка находятся где-то рядом,
# и они велики. Это говорит о том, что модель недообучилась: то есть она не имеет 
# достаточное кол-во параметров.

# Для сильно усложненных моделей тренировочная и валидационная ошибки значительно 
# отличаются. Это можно объяснить переобучением: когда параметров слишком много 
# либо не хватает регуляризации, алгоритм может "отвлекаться" на шум в данных и 
# упускать основной тренд.


						# Сколько нужно данных?

# Известно, что чем больше данных использует модель, тем лучше. Но как нам понять 
# в конкретной ситуации, помогут ли новые данные? Скажем, целесообразно ли нам 
# потратить N\$ на труд асессоров, чтобы увеличить выборку вдвое?

# Поскольку новых данных пока может и не быть, разумно поварьировать размер имеющейся 
# обучающей выборки и посмотреть, как качество решения задачи зависит от объема данных, 
# на котором мы обучали модель. Так получаются кривые обучения (learning curves).

# Идея простая: мы отображаем ошибку как функцию от количества примеров, используемых 
# для обучения. При этом параметры модели фиксируются заранее.

# Давайте посмотрим, что мы получим для линейной модели. Коэффициент регуляризации 
# выставим большим.

# def plot_learning_curve(degree=2, alpha=0.01):
#     train_sizes = np.linspace(0.05, 1, 20)
#     logit_pipe = Pipeline(
#         [
#             ("scaler", StandardScaler()),
#             ("poly", PolynomialFeatures(degree=degree)),
#             (
#                 "sgd_logit",
#                 SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha, max_iter=5),
#             ),
#         ]
#     )
#     N_train, val_train, val_test = learning_curve(
#         logit_pipe, X, y, train_sizes=train_sizes, cv=5, scoring="roc_auc"
#     )
#     plot_with_err(N_train, val_train, label="training scores")
#     plot_with_err(N_train, val_test, label="validation scores")
#     plt.xlabel("Training Set Size")
#     plt.ylabel("AUC")
#     plt.legend()
#     plt.grid(True);

# plot_learning_curve(degree=2, alpha=10)


# Типичная ситуация: для небольшого объема данных ошибки на обучающей выборке и в процессе 
# кросс-валидации довольно сильно отличаются, что указывает на переобучение. Для той же 
# модели, но с большим объемом данных ошибки "сходятся", что указывается на недообучение.

# Если добавить еще данные, ошибка на обучающей выборке не будет расти, но с другой стороны, 
# ошибка на тестовых данных не будет уменьшаться.

# Получается, ошибки "сошлись", и добавление новых данных не поможет. Собственно, это случай 
# – самый интересный для бизнеса. Возможна ситуация, когда мы увеличиваем выборку в 10 раз. 
# Но если не менять сложность модели, это может и не помочь. То есть стратегия "настроил один 
# раз – дальше использую 10 раз" может и не работать.

# Что будет, если изменить коэффициент регуляризации (уменьшить до 0.05)?
# plot_learning_curve(degree=2, alpha=0.05)

# Видим хорошую тенденцию – кривые постепенно сходятся, и если дальше двигаться направо 
# (добавлять в модель данные), можно еще повысить качество на валидации.

# А если усложнить модель ещё больше ($alpha=10^{-4}$)?
# plot_learning_curve(degree=2, alpha=1e-4)
# Проявляется переобучение – AUC падает как на обучении, так и на валидации.

# Строя подобные кривые, можно понять, в какую сторону двигаться, и как правильно настроить сложность модели на новых данных.

# Выводы по кривым валидации и обучения

# Ошибка на обучающей выборке сама по себе ничего не говорит о качестве модели
# Кросс-валидационная ошибка показывает, насколько хорошо модель подстраивается 
# под данные (имеющийся тренд в данных), сохраняя при этом способность обобщения 
# на новые данные
# Валидационная кривая представляет собой график, показывающий результат на 
# тренировочной и валидационной выборке в зависимости от сложности модели:
# если две кривые распологаются близко, и обе ошибки велики, — это признак 
# недообучения
# если две кривые далеко друг от друга, — это показатель переобучения
# Кривая обучения — это график, показывающий результаты на валидации и 
# тренировочной подвыборке в зависимости от количества наблюдений:
# если кривые сошлись друг к другу, добавление новых данных не поможет – 
# надо менять сложность модели
# если кривые еще не сошлись, добавление новых данных может улучшить результат.

						

						# Часть 6. Конкурс Kaggle Inclass «Поймай меня, если сможешь»
# Загрузка и преобразование данных
# Разреженные матрицы
# Обучение первой модели


# Будем решать задачу идентификации взломщика по его поведению в сети Интернет. Это 
# сложная и интересная задача на стыке анализа данных и поведенческой психологии. В 
# качестве примера, компания Яндекс решает задачу идентификации взломщика почтового 
# ящика по его поведению. В двух словах, взломщик будет себя вести не так, как владелец 
# ящика: он может не удалять сообщения сразу по прочтении, как это делал хозяин, он будет 
# по-другому ставить флажки сообщениям и даже по-своему двигать мышкой. Тогда такого 
# злоумышленника можно идентифицировать и "выкинуть" из почтового ящика, предложив хозяину 
# войти по SMS-коду. Этот пилотный проект описан в статье на Хабрахабре. Похожие вещи делаются, 
# например, в Google Analytics и описываются в научных статьях, найти можно многое по фразам 
# "Traversal Pattern Mining" и "Sequential Pattern Mining".

# В этом соревновании будем решать похожую задачу: алгоритм будет анализировать последовательность 
# из нескольких веб-сайтов, посещенных подряд одним и тем же человеком, и определять, Элис это или 
# взломщик (кто-то другой).

# Данные собраны с прокси-серверов Университета Блеза Паскаля. "A Tool for Classification of 
# Sequential Data", авторы Giacomo Kahn, Yannick Loiseau и Olivier Raynaud.

# В этом соревновании мы собираемся решить аналогичную задачу: наш алгоритм должен анализировать 
# последовательность веб-сайтов, последовательно посещаемых конкретным человеком, и предсказывать, 
# является ли этот человек пользователем по имени Алиса или злоумышленником (кем-то еще). В 
# качестве метрики мы будем использовать ROC AUC.

from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Загрузить наборы обучающих и тестовых данных
train_df = pd.read_csv('../cmiyc/train_sessions.csv', index_col='session_id')
test_df  = pd.read_csv('../cmiyc/test_sessions.csv',  index_col='session_id')

# Преобразование столбцов time1, ..., time10 в тип datetime
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Сортировать данные по времени
train_df = train_df.sort_values(by='time1')

# Посмотрите на первые строки обучающего набора
print(train_df.head())

# Набор данных для обучения содержит следующие функции:
# site1 - id первого посещенного в сеансе сайта;
# time1 - время посещения первого сайта в сеансе;
# …
# site10 - id десятого посещенного в сеансе сайта;
# time10 - время посещения десятого сайта за сессию;
# target - целевая переменная, принимает значение 1 для сеансов 
# Алисы и 0 для сеансов других пользователей.

# Пользовательские сеансы выбираются таким образом, чтобы они 
# длились не более получаса и / или содержали более десяти веб-сайтов; 
# то есть сеанс считается завершенным, если пользователь посетил десять 
# веб-сайтов или если сеанс длился более тридцати минут.

# В таблице есть несколько пустых значений, что означает, что некоторые 
# сеансы содержат менее десяти веб-сайтов. Замените пустые значения на 0 
# и измените типы столбцов на целые. Также загрузите словарь сайта и 
# посмотрите, как он выглядит:

# Измените тип столбцов site1, ..., site10 на целые и заполните NA-значения нулями
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Загрузить словарь веб-сайта
with open(r"../cmiyc/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Создать фрейм данных для словаря
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
print(sites_dict.head())

# Чтобы обучить нашу первую модель, нам нужно подготовить данные. Прежде всего, 
# исключите целевую переменную из обучающей выборки. Теперь и обучающий, и тестовый 
# наборы имеют одинаковое количество столбцов, и мы можем объединить их в один фрейм 
# данных. Таким образом, все преобразования будут выполняться одновременно как для 
# обучающих, так и для тестовых наборов данных.

# С одной стороны, это приводит к тому, что оба наших набора данных имеют одно 
# пространство объектов (поэтому вам не нужно беспокоиться о том, что вы, возможно, 
# забыли преобразовать объект в одном из наборов данных). С другой стороны, время 
# обработки увеличится. В случае очень больших наборов может оказаться, что невозможно 
# преобразовать оба набора данных одновременно (и иногда вам приходится разбивать 
# преобразования на несколько этапов, отдельно для набора данных train/test). В нашем 
# случае мы собираемся выполнить все преобразования для объединенного фрейма данных сразу; 
# и, прежде чем обучать модель или делать прогнозы, мы просто будем использовать 
# соответствующую ее часть.

# Наша целевая переменная
y_train = train_df['target']
# Единый фрейм исходных данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])
# Индекс для разделения наборов данных для обучения и тестирования
idx_split = train_df.shape[0]

# Для простоты мы будем использовать только посещенные веб-сайты в сеансе (и 
# мы не будем учитывать особенности временных меток). Смысл этого выбора данных 
# таков: у Алисы есть свои любимые сайты, и чем чаще вы видите эти сайты в сеансе, 
# тем выше вероятность того, что это сеанс Алисы, и наоборот.

# Подготовим данные. Мы сохраним во фрейме данных только функции site1, site2,…, site10.
# Имейте в виду, что отсутствующие значения были заменены нулями. Вот как выглядят 
# первые строки фрейма данных:

# Dataframe с индексами посещенных сайтов
full_sites = full_df[sites]
print(full_sites.head())

# Сеансы - это последовательности индексов сайта, и такое представление данных неудобно 
# для линейных методов. Согласно нашей гипотезе (у Алисы есть любимые веб-сайты), нам 
# необходимо преобразовать этот фрейм данных так, чтобы каждый веб-сайт имел соответствующую 
# функцию (столбец), значение которой равно количеству посещений этого веб-сайта в течение 
# сеанса. Все это можно сделать двумя строчками:

# последовательность индексов
sites_flatten = full_sites.values.flatten()
# и матрица, которую мы ищем
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0], sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

# Если вы понимаете, что здесь только что произошло, то можете пропустить следующий раздел 
# (возможно, вы тоже справитесь с логистической регрессией?). Если нет, то давайте разберемся.




plt.show()










