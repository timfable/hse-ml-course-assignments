import pandas as pd
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold

wine_data = pd.read_csv('wine.data', header=None, names=list('abcdefghijklmn'))

y = wine_data['a']
X = wine_data[list('bcdefghijklmn')]

print("Accuracy for K 1:50 without scaling: \n\n")

for k in range(1, 51):

    kf = KFold(len(wine_data), n_folds=5, shuffle=True, random_state=42)
    cls = KNeighborsClassifier(k)
    scores = cv.cross_val_score(cls, X, y, scoring='accuracy', cv=kf)

    print("K = %0i; Accuracy: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))

print("Accuracy for K 1:50 with scaling: \n\n")

X_scaled = sklearn.preprocessing.scale(X)

for k in range(1, 51):

    kf = KFold(len(wine_data), n_folds=5, shuffle=True, random_state=42)
    cls = KNeighborsClassifier(k)
    scores = cv.cross_val_score(cls, X_scaled, y, scoring='accuracy', cv=kf)

    print("K = %0i; Accuracy: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))
