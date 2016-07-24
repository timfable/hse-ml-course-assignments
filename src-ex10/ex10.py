import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

data_dj = pd.read_csv('djia_index.csv', header=0)
data_prices = pd.read_csv('close_prices.csv', header=0)

# data_prices['date'] = pd.to_datetime(data_prices['date'])
# data_prices['date'] = pd.to_numeric(data_prices['date'])

data_prices = data_prices.drop(['date'], axis=1)

pca = PCA(n_components=10)
pca.fit(data_prices)

print(pca.explained_variance_ratio_)

print(sum(pca.explained_variance_ratio_[0:4]))

X = pca.transform(data_prices)

print(pca.components_[0])

print(data_prices.columns[np.argmax(pca.components_[0])])

corr = np.corrcoef(X[:, 0], data_dj['^DJI'])

print(corr)
