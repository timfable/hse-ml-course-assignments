import pandas as pd
from sklearn.tree import DecisionTreeClassifier

titanic_cvs = pd.read_csv('titanic.csv', index_col='PassengerId')

df = pd.DataFrame(data=titanic_cvs, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
df = df.dropna(axis=0)
df['Sex'] = df['Sex'] == 'male'

X = pd.DataFrame(data=df, columns=['Pclass', 'Fare', 'Age', 'Sex'])
y = df['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
