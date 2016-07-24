import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from matplotlib import interactive


def sigm(y_pred):
    return 1 / (1 + math.exp(-y_pred))

df_data = pd.read_csv('gbm-data.csv', header=0)

np_data = df_data.values
X = np_data[:, 1:]
y = np_data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# interactive(True)
# plt.ion()

for lr in [1, 0.5, 0.3, 0.2, 0.1]:

    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)

    train_loss = np.zeros((250,), dtype=np.float64)
    test_loss = np.zeros((250,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        train_loss[i] = log_loss(y_train, list(map(sigm, y_pred)))

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_loss[i] = log_loss(y_test, list(map(sigm, y_pred)))

    i = np.argmin(test_loss)
    print(test_loss[i], i + 1)

    # plt.figure()
    # plt.plot(test_loss, 'r', linewidth=2)
    # plt.plot(train_loss, 'g', linewidth=2)
    # plt.legend(['test', 'train'])
    # plt.draw()
    # plt.show()
    # wait = input("PRESS ENTER TO CONTINUE.")

#plt.ioff()

clf = RandomForestClassifier(n_estimators=37, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
loss = log_loss(y_test, y_pred)

print(loss)
