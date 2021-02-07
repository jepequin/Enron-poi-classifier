#!/usr/bin/python

import pandas as pd
import numpy as np
import pickle 
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from pprint import pprint

### Input classifier name
print('-'*50)
valid_input = False
while not valid_input:
	classifier = input('Enter name of classifier ("SVC", "GaussianNB" or "RandomForest"): ')
	if classifier in ['SVC','GaussianNB','RandomForest']:
		valid_input = True
	else:
		print('Input is not valid')
print('-'*50)

### Extract dictionary from pickle file
## From help(pickle.load): *file* can be a binary file object opened for reading ("rb") 
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

### Select columns to use in dataframe 
financial_features = ['bonus','deferral_payments','deferred_income','director_fees','exercised_stock_options','expenses','loan_advances','long_term_incentive','other','restricted_stock','restricted_stock_deferred','salary','total_payments','total_stock_value']
email_features = ['from_poi_to_this_person','shared_receipt_with_poi','from_this_person_to_poi','from_messages','to_messages']
feature_list = financial_features + email_features
label_list = ['poi']
columns = feature_list + label_list 

### Create dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index', columns=columns)
## Replace string 'NaN' with float np.nan to avoid division errors below 
df = df.replace(to_replace='NaN',value=np.nan)

### Split dataframe into features and labels
features = df.loc[:,feature_list]
labels = df.loc[:,label_list]

### Create new features
## Feature 'percentage_shared_with_poi'
shared = df.loc[:,'shared_receipt_with_poi']
to_messages = df.loc[:,'to_messages']
features['percentage_shared_with_poi'] = shared.divide(to_messages)
## Feature 'percentage_sent_to_poi'
sent_to_poi = df.loc[:,'from_this_person_to_poi']  
from_messages = df.loc[:,'from_messages']
features['percentage_sent_to_poi'] = sent_to_poi.divide(from_messages)

### Replace non numerical values 
features.replace(to_replace=np.nan,value=0, inplace=True)
labels.replace(to_replace=False,value=0,inplace=True)
labels.replace(to_replace=True,value=1,inplace=True)

### Convert to numpy array 
features = features.to_numpy()
## Reshape labels array for compatibility with fit function below
labels = labels.to_numpy().reshape(len(labels))

### Select best features
print('Selecting best features ...')
feature_list = feature_list + ['percentage_shared_with_poi','percentage_sent_to_poi']
## Selecting five best features produces optimal accuracy, precision and recall
k = 5 
selector = SelectKBest(k=k)
selector.fit(features,labels)
best_features = selector.transform(features)

### Print selected features
mask = selector.get_support()
selected_features = []
for boolean, feature in zip(mask,feature_list):
	if boolean:
		selected_features.append(feature)
print('Selected {} features:'.format(k))
pprint(selected_features)
print('-'*50)

### Train/test split
features_train, features_test, labels_train, labels_test = \
    train_test_split(best_features, labels, stratify=labels, test_size=0.3, random_state=42)

### Search best parameters for a variety of classifiers
print('Selecting best parameters for {} classifier ...'.format(classifier))
if classifier == 'SVC':
	clf = SVC()
	parameters = {'kernel':('rbf','sigmoid'),'C':(0.1,1,10),'gamma':('scale','auto',0.1,1,10)}
if classifier == 'RandomForest':
	clf = RandomForestClassifier(random_state=0)
	parameters = {'n_estimators':(10,100,1000),'min_samples_split':(2,5,10),'max_depth':(10,50,100),'criterion':('gini','entropy')}
if classifier == 'GaussianNB':
	clf = GaussianNB()
	parameters = {'var_smoothing':(1e-08,1e-09,1e-10)} 
grid = GridSearchCV(clf, parameters, cv = 5, scoring = 'f1')
grid.fit(features_train,labels_train)
	
### Print best parameters
print('Best parameters:')
pprint(grid.best_params_)
print('-'*50)

### Pass best parameters to and train classifier
print('Feeding best parameters to model ...')
if classifier == 'SVC':
	clf = SVC()
if classifier == 'RandomForest':
	clf = RandomForestClassifier(random_state=0)
if classifier == 'GaussianNB':
	clf = GaussianNB()
clf.set_params(**grid.best_params_)
clf.fit(features_train,labels_train)

### Print metrics 
print('Accuracy of {} classifier: {}'.format(classifier,clf.score(features_test,labels_test)))
print('Precision of {} classifier: {}'.format(classifier,precision_score(clf.predict(features_test),labels_test)))
print('Recall of {} classifier: {}'.format(classifier,recall_score(clf.predict(features_test),labels_test)))
print('F1 score of {} classifier: {}'.format(classifier,f1_score(clf.predict(features_test),labels_test)))
print('-'*50)




