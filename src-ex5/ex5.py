import pandas as pd

from sklearn.svm import SVC

svm_data = pd.read_csv('svm-data.csv', header=None, names=list('abc'))

y = svm_data['a']
X = svm_data[list('bc')]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)

print(clf.support_)
