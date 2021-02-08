# ENRON PERSON OF INTEREST CLASSIFIER

This is the final project for the course "Intro to Machine Learning" by Udacity. This course, besides introducing some of the most widely used models in ML, presents some core steps of the machine learning workflow:

- Feature engineering: exploration, creation, selection, and transformation of features.
- Model selection and parameter tuning.
- Evaluation: cross validation, metric selection.

In this project we go through the end-to-end process of investigating a data set through a machine learning lens.

## DESCRIPTION

In this project we study one of the worst cases of corporate fraud: the Enron scandal. In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 

We aim at building a model that can predict whether an Enron employee is a "person of interest" or not.  A person of interest is anyone that was indicted, settled without admitting guilt or testified in exchange for immunity.

The data set for this project is in the pickle file "final_project_dataset.pkl". It contains a dictionary with Enron email and financial data, each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

### Financial features:

['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

### Email features:

['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

### POI label:

[‘poi’] (boolean)

A starter code is also provided. However, we choose not to use it and instead build the classifier from scratch. 

## USAGE

There are three models presented in the course: Decision Trees, Gaussian Naive Bayes, and Support vector classifiers. In this project we offer the user the possibility of choosing (input) one the following:

- Random Forest Classifier.
- Gaussian Naive Bayes. 
- Support Vector Classifier

