import pandas as pd
import numpy as np
import math

from sklearn.metrics import roc_auc_score


def cost(X, y, w1, w2):

    J = 0
    for i in range(0, len(X)):
        J += math.log(1 + math.exp(-y.loc[i] * (w1 * X.loc[i, 'b'] + w2 * X.loc[i, 'c'])))

    return J / len(X)


def cost_reg(X, y, w1, w2, C):
    J = cost(X, y, w1, w2)

    J += (C / 2) * (w1 ** 2 + w2 ** 2)

    return J


def grad(X, y, w1, w2, k):

    grad1 = 0
    grad2 = 0
    for i in range(0, len(X)):
        grad1 += y.loc[i] * X.loc[i, 'b'] * \
                 (1 - 1 / (1 + math.exp(-y.loc[i] * (w1 * X.loc[i, 'b'] + w2 * X.loc[i, 'c']))))
        grad2 += y.loc[i] * X.loc[i, 'c'] * \
                 (1 - 1 / (1 + math.exp(-y.loc[i] * (w1 * X.loc[i, 'b'] + w2 * X.loc[i, 'c']))))

    grad1 = grad1 * k / len(X)
    grad2 = grad2 * k / len(X)

    return [grad1, grad2]


def grad_reg(X, y, w1, w2, k, C):

    [grad1, grad2] = grad(X, y, w1, w2, k)

    grad1 += - k * C * w1
    grad2 += - k * C * w2

    return [grad1, grad2]


data_logistic = pd.read_csv('data-logistic.csv', header=None, names=list('abc'))

y = data_logistic['a']
X = data_logistic[list('bc')]

k = 0.1

w1 = 0
w2 = 0

grad1 = 1
grad2 = 1
i = 1

while abs(grad1) > 1e-5 and abs(grad2) > 1e-5 and i < 10000:

    J = cost(X, y, w1, w2)

    [grad1, grad2] = grad(X, y, w1, w2, k)

    w1 = w1 + grad1
    w2 = w2 + grad2

    i += 1

predict = np.zeros(len(y))
y_true = np.zeros(len(y))

for i in range(0, len(y)):
    y_true[i] = y.loc[i]
    predict[i] = 1 / (1 + math.exp(-w1 * X.loc[i, 'b'] - w2 * X.loc[i, 'c']))

score = roc_auc_score(y_true, predict)

print("%0.3f %0.3f" % (J, score))


C = 10

w1 = 0
w2 = 0

grad1 = 1
grad2 = 1
i = 1

while abs(grad1) > 1e-5 and abs(grad2) > 1e-5 and i < 10000:

    J = cost_reg(X, y, w1, w2, C)

    [grad1, grad2] = grad_reg(X, y, w1, w2, k, C)

    w1 = w1 + grad1
    w2 = w2 + grad2

    i += 1

predict = np.zeros(len(y))
y_true = np.zeros(len(y))

for i in range(0, len(y)):
    y_true[i] = y.loc[i]
    predict[i] = 1 / (1 + math.exp(-w1 * X.loc[i, 'b'] - w2 * X.loc[i, 'c']))

score = roc_auc_score(y_true, predict)

print("%0.3f %0.3f" % (J, score))
