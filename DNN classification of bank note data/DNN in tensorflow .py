# Data sourced from https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

bank = pd.read_csv('bank_note_data.csv')
bank.head()

# Countplot of the Classes (Authentic 1 vs Fake 0)
sns.set_style('darkgrid')
sns.countplot(data=bank, x='Class', palette='GnBu_r')

# PairPlot of the Data with Seaborn, Hue= Class
sns.pairplot(data=bank, hue='Class', palette='GnBu')

# Scale and fit
scaler = StandardScaler()
scaler.fit(bank.drop('Class', axis=1))
scaled_features = scaler.transform(bank.drop('Class', axis=1))
scaled_features = pd.DataFrame(scaled_features, columns=bank.columns[:-1])
scaled_features.head()

# Train Test Split
X = scaled_features
y = bank['Class']
X = X.as_matrix()
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create the DNNClassifier
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)

# Fit classifier to the training data.
classifier.fit(X_train, y_train, steps=200, batch_size=20)

# Evaluate the model
note_predictions = list(classifier.predict(X_test, as_iterable=True))

print(confusion_matrix(y_test, note_predictions))
print('\n')
print(classification_report(y_test, note_predictions))

# # Compared with random forest modeling
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
