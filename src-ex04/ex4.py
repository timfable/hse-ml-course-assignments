import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

perceptron_test = pd.read_csv('perceptron-test.csv', header=None, names=list('abc'))
perceptron_train = pd.read_csv('perceptron-train.csv', header=None, names=list('abc'))

y_test = perceptron_test['a']
X_test = perceptron_test[list('bc')]

y_train = perceptron_train['a']
X_train = perceptron_train[list('bc')]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %0.3f" % accuracy)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predictions)

print("Accuracy: %0.3f" % accuracy_scaled)

print("Accuracy: %0.3f" % (accuracy_scaled - accuracy))

