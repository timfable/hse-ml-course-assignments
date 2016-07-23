import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.datasets import load_boston

boston = load_boston()

X = sklearn.preprocessing.scale(boston.data)
y = boston.target

kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)

for p in np.linspace(1, 10, num=200):

    reg = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')

    scores = cv.cross_val_score(reg, X, y, scoring='mean_squared_error', cv=kf)

    print("p = %0.2f; Accuracy: %0.1f (+/- %0.2f)" % (p, scores.mean(), scores.std() * 2))
