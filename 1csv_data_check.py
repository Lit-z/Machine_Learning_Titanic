import pandas as pd

# a quick comparison of basic info of the data sets test won't be looked at
# again until testing the models produced by the work done on train data sets
train_df = pd.read_csv('titanic_train.csv')
test_df = pd.read_csv('titanic_test.csv')

print('\ntrain df info\n')
print(train_df.info())
print('\ntrain df describe\n')
print(train_df.describe())
print('\ntrain df head\n')
print(train_df.head())

print('\ntest df info\n')
print(test_df.info())
print('\ntest df describe\n')
print(test_df.describe())
print('\ntest df head\n')
print(test_df.head())

print('\ntest/ total data ratio:\n',len(test_df)/(len(train_df)+len(test_df)))

# from the above some info found out to compare them is that
# train data set has 891 entries while the test set has 418 entries
# ~32% of the total data is in the test data, the rest is in the training data
# format between the two types of data is similar