import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('titanic_train.csv')

print('\ntrain head\n')
print(train_df.info())

# from this info we can see that there are 12 columns of data, with missing data
# in Age (177/891 null), Cabin (687/891 null) and Embarked (2/891 null)
# can confirm this visually using a heatmap plot where empty data sections will
# appear as a different colour, via isnull()
plt.figure(figsize=(10,8))
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# looking at the head of the data
print('\ntrain head\n')
print(train_df.head())
# can identify that the PassengerId column is just index value +1

# checking for correlation between 'Sex' and 'Survived'
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.countplot(data=train_df,x='Survived',hue='Sex',palette='Set1')
plt.show()

# checking for correlation between 'Pclass' (passenger class, 1st, 2nd, 3rd)
# and 'Survived'
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.countplot(data=train_df, x='Survived',hue='Pclass',palette='Set2')
plt.show()
# these two previous plots show males and lower class passengers were less 
# likely to survive

# same idea for 'Embarked' and 'Survived'
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.countplot(data=train_df, x='Survived',hue='Embarked',palette='Set2')
plt.show()
# (S,C,Q) represent (Southampton, Cherbourg, Queenstown) which was the stops on
# the journey before departing across the Atlantic, ther does appear to be a 
# higher survival rate for passengers from C

# creating a df for the numerical fields and comparing them to if they survived
# For understanding the two not automatically clear titles, they mean who
# they were travelling with as a total where
# SibSp = siblings/ spouse, Parch = parents/ children
num_df = train_df[['Survived','Age','SibSp','Parch','Fare']]
sns.set_style('whitegrid')
sns.pairplot(num_df, hue='Survived')
plt.show()

# looking at each one individually
plt.figure(figsize=(18,4))
plt.subplot(1,4,1)
sns.histplot(data=train_df, x='Age', hue='Survived')

plt.subplot(1,4,2)
sns.histplot(data=train_df, x='Fare', hue='Survived')

plt.subplot(1,4,3)
sns.histplot(data=train_df, x='SibSp', hue='Survived')

plt.subplot(1,4,4)
sns.histplot(data=train_df, x='Parch', hue='Survived')
plt.show()
# Age: looks normally distributed, the highest survival rate for the youngest, 
# most deaths in the 20-30 range and poor survival rate for the oldest (even if
# the oldest individual looks like they survived)
# Fare: mostly cheaper fares, some extreme value around 500 that affects this 
# plot in terms of being good for readability but more expensive fares were
# likely to survive
# SibSp: no survival rate for larger groups of siblings/ spouses, 
# the majority of survivors and passengers overall were without sib/sp
# Parch: more likely to survive with 1-2 parents/ children but combined with
# SibSp data it seems many passengers were on their own