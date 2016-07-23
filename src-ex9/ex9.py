import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

data_train = pd.read_csv('salary-train.csv', header=0)
data_test = pd.read_csv('salary-test-mini.csv', header=0)

data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
X_train_desc = vectorizer.fit_transform(data_train['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_desc, X_train_categ])

clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, data_train['SalaryNormalized'])

data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
X_test_desc = vectorizer.transform(data_test['FullDescription'])
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = hstack([X_test_desc, X_test_categ])

predict = clf.predict(X_test)

print(predict)
