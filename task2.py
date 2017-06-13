from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import pandas as pd
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


dataset = pd.read_csv('./dataset_classification.csv', skiprows=1, names=[
    'N', 'Disability', 'Water resources', 'Adoption', 'Healthcare', 'Salvador', 'Religions',
    'Antisatelite', 'Nicarague assistance', 'Rocket', 'Immigrants', 'Alternative fuel',
    'Education', 'Funds', 'Crime', 'Customs', 'Exports', 'Class'
], delimiter=';')

# data normalization

mapping_dict = {
    'yes': 1,
    'no': 0,
    'abstain': 2,
    'republican' : 'republican',
    'democrat': 'democrat'
}

dataset = dataset.drop(['N'], axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)

for index, row in dataset.iterrows():
    for column in dataset:
        dataset[column][index] = mapping_dict[dataset[column][index]]


for column in dataset:
    if column != 'Class':
        dataset[column] = dataset[column].astype(int)

# splitting set to X and Y

'''X_dataset = dataset.drop(['Class'], axis=1)
Y_dataset = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, test_size=0.4, random_state=0)

# selecting features

model = LogisticRegression()
rfe = RFE(model, 1)
#rfe.fit(X_train, y_train)
#print(rfe.support_)
#print(rfe.ranking_)

print(X_train.columns)
X_train = X_train.drop('Immigrants', axis=1)
X_test = X_test.drop('Immigrants', axis=1)

# cross validation on selected features: disability, adoption, healthcare, nicarague, alternative fuel

enc = OneHotEncoder()
enc.fit(X_train)
OneHotEncoder(categorical_features='all',
       handle_unknown='error', n_values='auto')
X_train = enc.transform(X_train)
print(X_train)

cross_val = LogisticRegression().fit(X_train, y_train)
print(cross_val.score(enc.transform(X_test), y_test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.predict(enc.transform(X_test))

dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("congress.pdf")
print(cross_val.score(enc.transform(X_test), y_test))'''

X_dataset = dataset.drop(['Class'], axis=1)
Y_dataset = dataset['Class']

enc = OneHotEncoder()
enc.fit(X_dataset)
OneHotEncoder(categorical_features='all',
       handle_unknown='error', n_values='auto')
X_dataset = enc.transform(X_dataset).toarray()
print(enc.active_features_)
print(X_dataset)

X_train = []
X_test = []
X_validation = []
Y_train= []
Y_test = []
Y_validation = []

index = 0
for row in X_dataset:
    if index <= len(dataset.index)*0.5:
        X_train.append(row)
        Y_train.append(Y_dataset[index])
    if index > len(dataset.index)*0.5 and index < len(dataset.index)*0.8:
        X_test.append(row)
        Y_test.append(Y_dataset[index])
    if index >= len(dataset.index)*0.8:
        X_validation.append(row)
        Y_validation.append(Y_dataset[index])
    index += 1

#lr = LogisticRegression().fit(X_train, Y_train)
#predicted = lr.predict(X_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)
dot_data = tree.export_graphviz(clf, out_file=None, class_names=['democrat', 'republican'])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("congress.pdf")

print(Y_test)

index = 0
score = 0
for result in predicted:
    if result == Y_test[index]:
        score += 1
    index += 1

print(index)
print(score)

predicted_validation = clf.predict(X_validation)

index_val = 0
score_val = 0
for result in predicted_validation:
    if result == Y_validation[index_val]:
        score_val += 1
    index_val += 1

print(index_val)
print(score_val)
