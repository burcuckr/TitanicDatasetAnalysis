import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic/Titanic-Dataset.csv")
print(df.head(10))

print(df.describe())
print(df[df["Age"] > 60])

#Scatter Plot
plt.scatter(df["Survived"], df["Age"])
plt.title("Survived vs Age")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()

#Histogram
plt.hist(df["Pclass"], bins = 10)
plt.title("Pclass")
plt.xlabel("Pclass")
plt.ylabel("Values")
plt.show()

#Box Plot
plt.boxplot(df["Age"].dropna())
plt.show()

#correlation Heat Map
sns.heatmap(df.corr(numeric_only= True), annot = True)
plt.show()
