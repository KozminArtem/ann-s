						# https://www.kaggle.com/kashnitsky/a3-demo-decision-trees?scriptVersionId=14165135
										# Домашнее задание № 3 (демо).
											# Деревья решений

# Деревья решений с игрушечной задачей и набором данных для взрослых UCI

# Начнем с загрузки всех необходимых библиотек:

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import seaborn as sns
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ipywidgets import Image
from io import StringIO
import pydotplus #pip install pydotplus
import warnings
warnings.simplefilter('ignore')

				
				# Часть 1. Набор данных игрушек "Будут ли они? Не так ли?"

# Ваша цель - выяснить, как работают деревья решений, пройдя через игрушечную задачу. 
# Хотя одно дерево решений не дает выдающихся результатов, другие эффективные алгоритмы, 
# такие как повышение градиента и случайные леса, основаны на той же идее. Вот почему 
# может быть полезно знать, как работают деревья решений.

# Мы рассмотрим игрушечный пример бинарной классификации: человек А решает, пойдут ли 
# они на второе свидание с человеком Б. Это будет зависеть от его внешности, красноречия, 
# употребления алкоголя (только для примера) и количества денег потратил на первое свидание.				

# Создание набора данных

			# Создать фрейм данных с фиктивными переменными
# def create_df(dic, feature_list):
#     out = pd.DataFrame(dic)
#     out = pd.concat([out, pd.get_dummies(out[feature_list])], axis = 1)
#     out.drop(feature_list, axis = 1, inplace = True)
#     return out

# Некоторые значения функций присутствуют в тренировке и отсутствуют в тесте, и наоборот.
# def intersect_features(train, test):
#     common_feat = list( set(train.keys()) & set(test.keys()))
#     return train[common_feat], test[common_feat]

# features = ['Looks', 'Alcoholic_beverage','Eloquence','Money_spent']

					# Данные обучения

# df_train = {}
# df_train['Looks'] = ['handsome', 'handsome', 'handsome', 'repulsive',
#                          'repulsive', 'repulsive', 'handsome'] 
# df_train['Alcoholic_beverage'] = ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']
# df_train['Eloquence'] = ['high', 'low', 'average', 'average', 'low',
#                                    'high', 'average']
# df_train['Money_spent'] = ['lots', 'little', 'lots', 'little', 'lots',
#                                   'lots', 'lots']
# df_train['Will_go'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])

# df_train = create_df(df_train, features)
# print(df_train)

					# Данные испытаний

# df_test = {}
# df_test['Looks'] = ['handsome', 'handsome', 'repulsive'] 
# df_test['Alcoholic_beverage'] = ['no', 'yes', 'yes']
# df_test['Eloquence'] = ['average', 'high', 'average']
# df_test['Money_spent'] = ['lots', 'little', 'lots']
# # df_test = create_df(df_test, features)
# print(df_test)

# Некоторые значения функций присутствуют в тренировке и отсутствуют в тесте, и наоборот.
# y = df_train['Will_go']
# df_train, df_test = intersect_features(train=df_train, test=df_test)
# print(df_train)
# print(df_test)

# Нарисуйте дерево решений (вручную или в любом графическом редакторе) для этого набора данных. 
# При желании вы также можете реализовать построение дерева и нарисовать его здесь.

# print('1. Какова энтропия S0 исходной системы? Под состояниями системы мы понимаем ')
# print('значения бинарного признака «Will_go» - 0 или 1 - всего два состояния.')
# S_0 = -3/7*math.log2(3/7) - 4/7*math.log2(4/7)
# print('S_0 = - 3/7log_2(3/7) - 4/7log_2(4/7) = ' + str(S_0))

# print('2. Разделим данные по признаку "Looks_handsome". Что такое энтропия S1')
# print('левой группы - та, у которой есть "Looks_handsome". Что такое энтропия S2')
# print('в противоположной группе? Каков информационный прирост (IG), если мы рассмотрим такой раскол?')
# S_1 = -1/4*math.log2(1/4) - 3/4*math.log2(3/4)
# S_2 = -2/3*math.log2(2/3) - 1/3*math.log2(1/3)
# print('S_1 = ' + str(S_1))
# print('S_2 = ' + str(S_2))
# IG = S_0-4/7*S_1-3/7*S_2
# print('IG = S_0 - 4/7*S_1-3/7*S_2 = ' + str(IG))

# Обучите дерево решений, используя sklearn на обучающих данных. Вы можете выбрать любую глубину дерева.
# Дополнительно: отобразить получившееся дерево с помощью graphviz. 
# Вы можете использовать pydot или веб-сервис dot2png.

# clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
# clf_tree.fit(df_train, y)

# dot_data = StringIO()
# export_graphviz(clf_tree, feature_names = list(set(df_train.keys())) , out_file=dot_data, filled=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(value = graph.create_png())																	
# graph.write_pdf("Tree_Assignment/Tree_1.pdf")
# plt.figure(figsize=(10, 6))
# plot_tree(clf_tree, feature_names = df_train.columns, filled=True ,class_names=["Won't go", "Will go"]);

# Часть 2. Функции расчета энтропии и прироста информации.

# Рассмотрим следующий пример разминки: у нас 9 синих мячей и 11 желтых мячей. 
# Пусть мяч имеет метку 1, если он синий, и 0 в противном случае.

# balls = [1 for i in range(9)] + [0 for i in range(11)]

# Затем разделите шары на две группы:

# balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
# balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow

# def entropy(a_list):
#     lst = list(a_list)
#     size = len(lst) 
#     entropy = 0
#     set_elements = len(set(lst))
#     if set_elements in [0, 1]:
#         return 0
#     for i in set(lst):
#         occ = lst.count(i)
#         entropy -= occ/size * math.log (occ/size,2)
#     return entropy

# print(entropy(balls)) # 9 blue, 11 yellow
# print(entropy(balls_left)) # 8 blue, 5 yellow
# print(entropy(balls_right)) # 1 blue, 6 yellow
# print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die

# print('Какова энтропия состояния, заданного списком balls_left?')
# print(entropy(balls_left)) # 8 blue, 5 yellow

# print('Какова энтропия честных игральных костей? (где мы смотрим на кости как на систему с 6 равновероятными состояниями)?')
# print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die

# def information_gain(root, left, right):
#     l_root = len(root)
#     l_right = len(right)
#     l_left = len(left)
#     IG = entropy(root) - 1.0*(l_left)/l_root*entropy(left) - 1.0*(l_right)/l_root*entropy(right) 
#     return IG

# print('Какую пользу дает разделение исходного набора данных на balls_left и balls_right?')
# print(information_gain(balls, balls_left, balls_right))

# Выводит прирост информации при разбиении по лучшему признаку
# def information_gains(X, y):
#     out = []
#     for i in X.columns:
#         out.append(information_gain(y, y[X[i] == 0], y[X[i] == 1]))
#     return out    

# print(information_gains(df_train, y))

# По желанию:
# Реализуйте алгоритм построения дерева решений, рекурсивно вызывая best_feature_to_split
# Постройте получившееся дерево

# def btree(X, y, feature_names):
#     clf = information_gains(X, y)
#     best_feat_id = clf.index(max(clf))
#     best_feature = feature_names[best_feat_id]
#     print (f'Best feature to split: {best_feature}')
    
#     x_left = X[X.iloc[:, best_feat_id] == 0]
#     x_right = X[X.iloc[:, best_feat_id] == 1]
#     print (f'Samples: {len(x_left)} (left) and {len(x_right)} (right)')
    
#     y_left = y[X.iloc[:, best_feat_id] == 0]
#     y_right = y[X.iloc[:, best_feat_id] == 1]
#     entropy_left = entropy(y_left)
#     entropy_right = entropy(y_right)
#     print (f'Entropy: {entropy_left} (left) and {entropy_right} (right)')
#     print('_' * 30 + '\n')
#     if entropy_left != 0:
#         print(f'Splitting the left group with {len(x_left)} samples:')
#         btree(x_left, y_left, feature_names)
#     if entropy_right != 0:
#         print(f'Splitting the right group with {len(x_right)} samples:')
#         btree(x_right, y_right, feature_names)

# print(btree(df_train, y, df_train.columns))

                        
                                    # Часть 3. Набор данных "для взрослых"

                                # Описание набора данных:
# Набор данных UCI для взрослых (не нужно загружать его, у нас есть копия в репозитории курса): 
# классифицируйте людей с использованием демографических данных - независимо от того, зарабатывают 
# они более 50 000 долларов в год или нет.


# Feature descriptions:

# Age – continuous feature
# Workclass – continuous feature
# fnlwgt – final weight of object, continuous feature
# Education – categorical feature
# Education_Num – number of years of education, continuous feature
# Martial_Status – categorical feature
# Occupation – categorical feature
# Relationship – categorical feature
# Race – categorical feature
# Sex – categorical feature
# Capital_Gain – continuous feature
# Capital_Loss – continuous feature
# Hours_per_week – continuous feature
# Country – categorical feature

# Target – earnings level, categorical (binary) feature.

pd.set_option('display.max_columns', 100)

data_train = pd.read_csv('../mlcourse.ai/data/adult_train.csv', sep = ';')
# print(data_train.tail())
data_test = pd.read_csv('../mlcourse.ai/data/adult_test.csv', sep = ';')
# print(data_test.tail())
# print(data_test['Target'].value_counts())
# необходимо удалить строки с неправильными метками в тестовом наборе данных
data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target']==' <=50K.')]
# print(data_test['Target'].value_counts())

# кодируем целевую переменную как целое число
data_train.loc[data_train['Target']==' <=50K', 'Target'] = 0
data_train.loc[data_train['Target']==' >50K', 'Target'] = 1
data_train['Target'] = data_train['Target'].astype('int64')




data_test.loc[data_test['Target']==' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target']==' >50K.', 'Target'] = 1
# print(data_test['Target'].value_counts())
data_test['Target'] = data_test['Target'].astype('int64')
# Первичный анализ данных
# print(data_test.describe(include='all').T)
# print(data_train['Target'].value_counts())

# fig = plt.figure(figsize=(20, 10))
# cols = 5
# rows = int(np.ceil(float(data_train.shape[1]) / cols))

# for i, column in enumerate(data_train.columns):
#     ax = fig.add_subplot(rows, cols, i + 1)
#     ax.set_title(column)
#     if data_train.dtypes[column] == np.object:
#         data_train[column].value_counts().plot(kind="bar", axes=ax)
#     else:
#         data_train[column].hist(axes=ax)
#         plt.xticks(rotation="vertical")
# plt.subplots_adjust(hspace=0.7, wspace=0.2)

# Проверка типов данных
# print(data_train.dtypes)
# print(data_test.dtypes)
# Как мы видим, в тестовых данных возраст трактуется как объект типа. Нам нужно это исправить.

data_test['Age'] = data_test['Age'].astype(int)

# Также мы приведем все функции float к типу int, чтобы типы данных 
# нашего поезда и тестовых данных были согласованными.

data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)

# Заполните недостающие данные для непрерывных объектов их средними 
# значениями, для категориальных объектов - их режимом.

# print(data_train.info())

# мы видим пропущенные значения

categorical_columns = [c for c in data_train.columns 
                       if data_train[c].dtype.name == 'object']
numerical_columns = [c for c in data_train.columns 
                     if data_train[c].dtype.name != 'object']

# print('categorical_columns:', categorical_columns)
# print('numerical_columns:', numerical_columns)

# заполнить недостающие данные

for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode()[0], inplace=True)
    data_test[c].fillna(data_train[c].mode()[0], inplace=True)
    
for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)

# больше нет пропущенных значений
# print(data_train.info ())

# Мы создадим фиктивный код для некоторых категориальных характеристик: рабочий класс, 
# образование, боевой_статус, род занятий, отношения, раса, пол, страна. Это можно 
# сделать с помощью метода pandas get_dummies

data_train = pd.concat([data_train[numerical_columns], pd.get_dummies(data_train[categorical_columns])], axis=1)

data_test = pd.concat([data_test[numerical_columns], pd.get_dummies(data_test[categorical_columns])], axis=1)

# print(set(data_train.columns) - set(data_test.columns))
# print(data_train.shape, data_test.shape)

# В тестовых данных нет Голландии. Создайте новый объект с нулевым значением.

data_test['Country_ Holand-Netherlands'] = 0
# print(set(data_train.columns) - set(data_test.columns))
# print(data_train.head(2))
# print(data_test.head(2))

X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

                                
                                # 3.1 Дерево решений без настройки параметров

# Обучите дерево решений (DecisionTreeClassifier) с максимальной глубиной 3 и оцените 
# метрику точности на тестовых данных. Используйте параметр random_state = 17 
# для воспроизводимости результатов.

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)

# dot_data = StringIO()
# export_graphviz(tree, feature_names = X_train.columns, out_file=dot_data, filled=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(value = graph.create_png())                                                                 
# graph.write_pdf("Tree_Assignment/Tree_2.pdf")

# Сделайте прогноз с помощью обученной модели на тестовых данных.
tree_predictions = tree.predict(X_test) 
# print(type(tree_predictions))
# Какова точность набора тестов для дерева решений с максимальной глубиной дерева 3 и random_state = 17?
print(accuracy_score(y_test, tree_predictions))

# Обучите дерево решений (DecisionTreeClassifier, random_state = 17). Найдите оптимальную 
# максимальную глубину с помощью 5-кратной перекрестной проверки (GridSearchCV).

tree_params = {'max_depth': range(2,11)}

locally_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17),tree_params, cv=5)
locally_best_tree.fit(X_train, y_train)

print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

# Обучите дерево решений с максимальной глубиной 9 (в моем случае это лучший max_depth) и 
# вычислите точность тестового набора.Используйте параметр random_state = 17 для воспроизводимости.

tuned_tree = DecisionTreeClassifier(max_depth = locally_best_tree.best_params_['max_depth'], random_state=17)
tuned_tree.fit(X_train, y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
print('7. Какова точность набора тестов для дерева решений с максимальной глубиной 9 и random_state = 17?')
print(accuracy_score(y_test, tuned_tree_predictions))

						# 3.3 (Необязательно) Случайный лес без настройки параметров

# Давайте взглянем на предстоящие лекции и попробуем использовать случайный лес для нашей задачи. 
# А пока вы можете представить себе случайный лес как группу деревьев решений, обученных на 
# немного разных подмножествах обучающих данных.

# Обучить случайный лес (RandomForestClassifier). Установите количество деревьев на
# 100 и используйте random_state = 17.

rf = RandomForestClassifier(n_estimators=100, random_state=17)
rf.fit(X_train,y_train)

# Сделайте прогнозы на основе тестовых данных и оцените точность.

# Выполните перекрестную проверку.

# %%time
cv_scores = cross_val_score(rf, X_train, y_train, cv=3)

print(cv_scores, cv_scores.mean())

forest_predictions = rf.predict(X_test) 
print(accuracy_score(y_test,forest_predictions))

						# 3.4 (Необязательно) Случайный лес с настройкой параметров

# Обучить случайный лес (RandomForestClassifier). Настройте максимальную глубину и 
# максимальное количество функций для каждого дерева с помощью GridSearchCV.


forest_params = {'max_depth': range(10, 21), 'max_features': range(5, 105, 20)}

locally_best_forest = GridSearchCV(RandomForestClassifier(n_estimators=10, random_state=17,n_jobs=4),forest_params, cv=3, verbose=1, n_jobs=4)
locally_best_forest.fit(X_train, y_train)

print("Best params:", locally_best_forest.best_params_)
print("Best cross validaton score", locally_best_forest.best_score_)

tuned_forest_predictions = locally_best_forest.predict(X_test) 
print('Сделайте прогнозы на основе тестовых данных и оцените точность.')
print(accuracy_score(y_test,tuned_forest_predictions))

# Добро пожаловать на 4-ю неделю нашего курса! Теперь мы представим нашу самую важную тему - линейные модели. 
# Если у вас есть подготовленные данные и вы хотите приступить к обучению моделей, то, скорее всего, вы сначала 
# попробуете либо линейную, либо логистическую регрессию, в зависимости от вашей задачи (регрессия или классификация).
# Материал этой недели охватывает как теорию линейных моделей, так и практические аспекты их использования в реальных 
# задачах. В этой теме будет много математики, и мы даже не будем пытаться отображать все формулы на Medium. Вместо 
# этого мы предоставляем блокнот Jupyter для каждой части этой статьи. В этом задании вы пройдете два простых теста 
# в конкурсе Kaggle, решив задачу идентификации пользователя на основе его сеанса посещенных веб-сайтов.

plt.show()

















