# https://nbviewer.jupyter.org/github/Yorko/mlcourse.ai/blob/master/jupyter_russian/assignments_demo/assignment01_adult_pandas.ipynb
										# Домашнее задание № 1 (демо).
								# Анализ данных по доходу населения UCI Adult


# Уникальные значения признаков 
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# salary: >50K,<=50K

import pandas as pd
import numpy as np

data = pd.read_csv("../mlcourse.ai/data/adult.data.csv")
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)
# print(data.head(2))
# print(data.info())
# print(data.describe())
# print(data.describe(include=['object', 'bool']))
print('1) How many men and women (sex feature) are represented in this dataset?')
print('Answ:')
print(data['sex'].value_counts())
# print(data['sex'].value_counts(normalize=True))

print('2) What is the average age (age feature) of women?')
print('Answ:')
print(data[data['sex'] == 'Female']['age'].mean())

print('3) What is the percentage of German citizens (native-country feature)?')
print('Answ:')
# print(type(data['native-country']))
# print(data['native-country'])
# print(type(data['native-country'].value_counts()))
# print(data['native-country'].value_counts())
# print(type(data['native-country'].value_counts(normalize=True)['Germany']))
print(data['native-country'].value_counts(normalize=True)['Germany'])

print('4,5) What are the mean and standard deviation of age for those who earn more')
print('than 50K per year (salary feature) and those who earn less than 50K per year?')
print('Answ:')
print('Mean <=50K: ' + str(data[data['salary'] == '<=50K']['age'].mean()))
print('Std  <=50K: ' + str(data[data['salary'] == '<=50K']['age'].std()))
print('Mean  >50K: ' + str(data[data['salary'] == '>50K']['age'].mean()))
print('Std   >50K: ' + str(data[data['salary'] == '>50K']['age'].std()))

print('6) Is it true that people who earn more than 50K have at least high school education?') 
print('(education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)')
# print(data['education'].value_counts(normalize=True))
print('Answ:')
print(data[data['salary'] == '>50K']['education'].value_counts())

print('7) Display age statistics for each race (race feature) and each gender (sex feature).')
print('Use groupby() and describe(). Find the maximum age of men of Amer-Indian-Eskimo race.')
print('Answ:')
print(data.groupby(['race','sex'])['age'].describe())
temp1 = data.groupby(['race','sex'])['age'].describe()
print('Maximum age of men of Amer-Indian-Eskimo race: ' +  str(temp1.loc[('Amer-Indian-Eskimo', 'Male'),'max']))

print('8) Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)?')
print('Consider as married those who have a marital-status starting with Married (Married-civ-spouse,')
print('Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.')
print('Answ:')
print('married man >50K:')
print(data[data['marital-status'].str.startswith('Married')]['salary'].value_counts(normalize=True)['>50K'])
print('single men >50K:')
print(data[~(data['marital-status'].str.startswith('Married'))]['salary'].value_counts(normalize=True)['>50K'])
# print(pd.crosstab(data['marital-status'], data['salary'], normalize = True, margins=True))

print('9) What is the maximum number of hours a person works per week (hours-per-week feature)? How many people')
print('work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?')
print('Answ:')
temp2 = data['hours-per-week'].max()
print('Maximum number of hours a person works per week: ' + str(temp2))
# print(data[data['hours-per-week'] == temp2].describe())
temp3 = data[data['hours-per-week'] == temp2]
print('People work such a number of hours: ' + str(temp3.shape[0]))
print('Percentage of those who earn a lot (>50K): '+ str(temp3['salary'].value_counts(normalize=True)['>50K']))

print('10) Count the average time of work (hours-per-week) for those who earn a little and a lot (salary)')
print('for each country (native-country). What will these be for Japan?')
print('Answ:')
temp4 = data.groupby(['native-country','salary'])['hours-per-week'].mean()
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print(temp4)
print('What will these be for Japan?')
print(temp4.loc['Japan'])





