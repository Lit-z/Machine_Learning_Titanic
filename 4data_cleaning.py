import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting up the df's
train_df = pd.read_csv('titanic_train.csv')
num_df = train_df[['Survived','Age','SibSp','Parch','Fare']]

# from previous stuff we know there is some missing age data, a lot of
# missing cabin data and two cases of missing embarked data
# to handle the missing age data will try estimating ages using other data

plt.figure(figsize=(12,6))
sns.boxplot(data=train_df,x='Pclass',y='Age',hue='Sex',palette='Set1')
plt.show()
# from this can see age increases with lower class (numerically, not wording) 
# and males are slightly older than females in all classes

# possible mean vs median exploration thoughts for later
# print(np.mean(train_df[(train_df['Pclass']==1) &
#                       (train_df['Sex']=='male')]['Age']))

# using the medians from the graph will be Pclass = 1, male 40, female 35
# Pclass = 2, male 30, female 28 and Pclass = 3, male 25, female 22

# creating a function that will put in the above median ages where the age
# value is missing
def age_estimate(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]

    if pd.isnull(Age):
        if Pclass == 1:
            if Sex == 'male':
                return 40
            else:
                return 35
        elif Pclass == 2:
            if Sex == 'male':
                return 30
            else:
                return 28
        else:
            if Sex == 'male':
                return 25
            else:
                return 22
    else:
        return Age

# calling the function to update the missing ages through apply()
train_df['Age'] = train_df[['Age','Pclass','Sex']].apply(age_estimate, axis=1)

# checking these updated ages on the isnull() heat map
plt.figure(figsize=(10,8))
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# dropping the cabin column due to it mostly being incomplete data
# could come back to this later
train_df.drop('Cabin',axis=1,inplace=True)

# finally dropping the two incomplete entries for Embarked info
train_df.dropna(inplace=True)

# converting categorical features into numerical using pd.get_dummies
gender = pd.get_dummies(train_df['Sex'],drop_first=True)
embark = pd.get_dummies(train_df['Embarked'],drop_first=True)

# removing the columns that are no longer needed and other info that doesn't 
# help with the current model aims such as Name and Ticket
# also removal of Passenger Id as it's just the same as an Index column
train_df.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,
                inplace=True)
# adding the replacement numerical columns
train_df = pd.concat([train_df,gender,embark],axis=1)

# saving the modified trainining data with these columsn to a new file for
# the next step
train_df.to_csv('modified_train.csv',index=False)