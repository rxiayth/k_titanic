import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic_df = pd.read_csv("train.csv")

# print("Data Shape \n", titanic_df.shape, "\n--------------------------------\n\n")
# print("Data Columns \n", titanic_df.columns, "\n--------------------------------\n\n")
# print("Data Info \n", titanic_df.info(), "\n--------------------------------\n\n")
# print("Data Describe \n", titanic_df.describe(), "\n--------------------------------\n\n")
# print("Data Head \n", titanic_df.head(), "\n--------------------------------\n\n")
# print("Data Tail \n", titanic_df.tail(), "\n--------------------------------\n\n")


# print("Data Value Counts - Survived \n", titanic_df["Survived"].value_counts(dropna=False), "\n--------------------------------\n\n")
# print("Data Value Counts - Age \n", titanic_df["Age"].value_counts(dropna=False), "\n--------------------------------\n\n")
# print("Data Value Counts - Pclass \n", titanic_df["Pclass"].value_counts(dropna=False), "\n--------------------------------\n\n")



# age_df = titanic_df["Age"]
# age_df = age_df.dropna()
# print(age_df)

# plt.hist(age_df)
# plt.show()
# plt.clf()


# count those under 16 and survived
children = titanic_df["Age"] <= 16
survied = titanic_df["Survived"] == 1
deceased = titanic_df["Survived"] == 0

children_survived = np.logical_and(children, survied)
children_survived_df = titanic_df[children_survived]

children_deceased = np.logical_and(children, deceased)
children_deceased_df = titanic_df[children_deceased]
# print(children_survived_df[["PassengerId", "Age", "Survived"]].sort_values(by="Age", ascending=True))
# print(children_deceased_df[["PassengerId", "Age", "Survived"]].sort_values(by="Age", ascending=True))
# plt.hist(children_survived)
# plt.show()
# plt.clf()

# titanic_df.boxplot(column="Age", by="Sex")
# print(titanic_df["Age"])
# print(titanic_df["Fare"])

plt.plot(kind="scatter", x= titanic_df["Age"], y=titanic_df["Fare"])

plt.show()


