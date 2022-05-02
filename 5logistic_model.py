from distutils.log import Log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# setting up the df's
train_df = pd.read_csv('modified_train.csv')

print(train_df.head())
# checking the data, can see the male, Q & S numeric columns of 0, 1's that 
# represent the old sex and embark columns if both Q & S are 0 it implies C
# was the embark location

# creating a training data set
x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived',
                axis=1), train_df['Survived'], test_size=0.4)

from sklearn.linear_model import LogisticRegression

# producing a logistic model from the training data
# making predictions for the testing data
lmodel = LogisticRegression(max_iter=500)
lmodel.fit(x_train,y_train)
pred = lmodel.predict(x_test)

# producing the classification report and confusion matrix to see the various
# measures of accuracy for the results
from sklearn.metrics import classification_report, confusion_matrix
print('\nclassification report for logistic model\n')
print(classification_report(y_test,pred))
print('\nconfusion matrix for logistic model\n')
print(confusion_matrix(y_test,pred))