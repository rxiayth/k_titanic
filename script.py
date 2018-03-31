import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv", sep=",")
test = pd.read_csv("test.csv")

# explore
        

        # print("Data Value Counts - Survived \n", train["Survived"].value_counts(dropna=False), "\n--------------------------------\n\n")
        # print("Data Value Counts - Age \n", train["Age"].value_counts(dropna=False), "\n--------------------------------\n\n")
        # print("Data Value Counts - Pclass \n", train["Pclass"].value_counts(dropna=False), "\n--------------------------------\n\n")


        # age_df = train["Age"]
        # age_df = age_df.dropna()
        # print(age_df)

        # plt.hist(age_df)
        # plt.show()
        # plt.clf()


        # # count those under 16 and survived
        # children = train["Age"] <= 16
        # survied = train["Survived"] == 1
        # deceased = train["Survived"] == 0

        # children_survived = np.logical_and(children, survied)
        # children_survived_df = train[children_survived]

        # children_deceased = np.logical_and(children, deceased)
        # children_deceased_df = train[children_deceased]
        # # print(children_survived_df[["PassengerId", "Age", "Survived"]].sort_values(by="Age", ascending=True))
        # # print(children_deceased_df[["PassengerId", "Age", "Survived"]].sort_values(by="Age", ascending=True))
        # # plt.hist(children_survived)
        # # plt.show()
        # # plt.clf()

        # # train.boxplot(column="Age", by="Sex")
        # # print(train["Age"])
        # # print(train["Fare"])

        # plt.plot(kind="scatter", x= train["Age"], y=train["Fare"])

        # plt.show()

## Explore Data
print('\x1b[6;30;42m' + "1. Explore Data" + '\x1b[0m' + '\n')
# print("Data Shape \n", train.shape, "\n--------------------------------\n\n")
# print("Data Columns \n", train.columns, "\n--------------------------------\n\n")
# print("Data Info \n", train.info(), "\n--------------------------------\n\n")
# print("Data Describe \n", train.describe(), "\n--------------------------------\n\n")
# print("Data Head \n", train.head(), "\n--------------------------------\n\n")
# print("Data Tail \n", train.tail(), "\n--------------------------------\n\n")
# print("check passengers survived vs dead via count and then %")
# print(train['Survived'].value_counts())
# print(train['Survived'].value_counts(normalize = True))

# print("check males that survived vs dead via count and then %") 
# print(train['Survived'][train['Sex'] == 'male'].value_counts())
# print( train['Survived'][train['Sex']== 'male'].value_counts(normalize = True))

# print("check females that survived vs dead via count and then %")
# print(train['Survived'][train['Sex'] == 'female'].value_counts())
# print( train['Survived'][train['Sex']== 'female'].value_counts(normalize = True))

## Clean Data
print('\x1b[6;30;42m' + "2. Clean Data" + '\x1b[0m' + '\n')
# print("create new Child (age<18) column")
# train["Child"] = float('NaN')
# train["Child"][train['Age'] < 18] = 1
# train["Child"][train['Age'] >= 18] = 0
# print(train.loc[0:10, ["Age", "Child"]])
# print("note Child col has missing values because Age has nan")
train["Age"] = train["Age"].fillna(train["Age"].median())
# print(train["Age"].describe())


# print("check survival rates for child, non-child, and all")
# print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
# print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
# print(train["Survived"][train["Child"]].value_counts(normalize = True))

# Decision Tree
print('\x1b[6;30;42m' + "3. Model Data" + '\x1b[0m' + '\n')
# print("convert categorical data to numerical format for Sex")
train["SexNum"] = int(0)
train["SexNum"][train["Sex"] == "male"] = 0
train["SexNum"][train["Sex"] == "female"] = 1
# print(train.loc[0:10, ["Sex", "SexNum"]])

# print("impute Embarked data, and then convert to numerical format")
train["Embarked"] = train["Embarked"].fillna("S")
train["EmbarkedNum"] = int(0)
train["EmbarkedNum"][train["Embarked"] == "S"] = 0
train["EmbarkedNum"][train["Embarked"] == "C"] = 1
train["EmbarkedNum"][train["Embarked"] == "Q"] = 2
# print (train.loc[0:10, ["Embarked",'EmbarkedNum']])


# print("select and create target and features numpy arrays for model")
target = train['Survived'].values
features_col = ["Pclass", "SexNum", "Age", "Fare"]
features_one = train[features_col].values
# print(train[["Pclass", "SexNum", "Age", "Fare"]].describe())

# print("fit decision tree")
my_tree_one = tree.DecisionTreeClassifier().fit(features_one,target)
# print("check importance and score of included features")
for i, feature in enumerate(features_col):
    print( str(feature) + " : " + str(my_tree_one.feature_importances_[i]))
# print("feature score: " + str(my_tree_one.score(features_one, target)))


print("Apply cleaning in test data")
# print(test.describe())
# print("note that Fare is missing data")
null_data = test[test["Fare"].isnull()]
# print(null_data)
test.Fare[152] = test.Fare.median()
test['Age'] = test['Age'].fillna(test['Age'].median())
test["SexNum"] = int(0)
test["SexNum"][test["Sex"] == "male"] = 0
test["SexNum"][test["Sex"] == "female"] = 1
print(test.describe())

print("replicate training model in test")
test_features = test[['Pclass', 'SexNum', 'Age', 'Fare']].values
my_prediction = my_tree_one.predict(test_features)
# print(my_prediction)

print("create a data frame with two columns: PassengerId & Survived")
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution.shape)

my_solution.to_csv("solution_one.csv", index_label = ["PassengerId"])
