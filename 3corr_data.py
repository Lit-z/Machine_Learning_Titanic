import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting up the df's
train_df = pd.read_csv('titanic_train.csv')
num_df = train_df[['Survived','Age','SibSp','Parch','Fare']]

# producing a heat map to check the strenght of correlation between the 
# numerical fields (num_df)
plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(),cmap='coolwarm',annot=True,lw=1,linecolor='black')
plt.show()
# initial correlation check has a moderate positive one between SibSp and Parch
# and a weak positive one between Parch and Fare
# while Age seems to be lacking a correlation with Survived

