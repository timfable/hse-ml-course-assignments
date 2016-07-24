import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

data = pd.read_csv('abalone.csv', header=0)

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data
X = X.drop('Rings', axis=1)
y = data['Rings']

for n in range(1, 51):

    kf = KFold(len(X), n_folds=5, shuffle=True, random_state=1)
    cls = RandomForestRegressor(n_estimators=n, random_state=1)
    scores = cross_val_score(cls, X, y, scoring='r2', cv=kf)

    print("n = %d; Accuracy: %0.2f (+/- %0.2f)" % (n, scores.mean(), scores.std() * 2))
