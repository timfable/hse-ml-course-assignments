import numpy as np

from operator import itemgetter
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

max_score = 0
max_C = 0
for a in gs.grid_scores_:
    if a.mean_validation_score > max_score:
        max_score = a.mean_validation_score
        max_C = a.parameters['C']

clf = SVC(C=max_C, kernel='linear', random_state=241)
clf.fit(X, y)

svm_coef = abs(clf.coef_).toarray()

top10 = sorted(zip(svm_coef[0], feature_mapping), key=itemgetter(0))[-10:]

for coef, feat in top10:
    print(feat, coef)

print("\n")

top10 = sorted(top10, key=itemgetter(1))

for coef, feat in top10:
    print(feat, coef)
