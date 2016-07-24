# HSE Machine Learning Coursera Course - Assignments

## ex01 - Важность признаков

* Поиск решающих деревьев
* Поиск наиболее важных признаков

```python
sklearn.tree.DecisionTreeСlassifier
```

## ex02 - Выбор числа соседей

* Метод k ближайших соседей, выбор параметра k
* Подготовка данных к использованию в методе kNN

```python
sklearn.neighbors.KNeighborsClassifier

sklearn.cross_validation.cross_val_score
sklearn.cross_validation.KFold

sklearn.preprocessing.scale
```

## ex11 - Размер случайного леса

* Работа со случайным лесом
* Решение с его помощью задачи регрессии
* Подбор параметров случайного леса

```python
sklearn.ensemble.RandomForestClassifier
sklearn.ensemble.RandomForestRegressor

sklearn.cross_validation.cross_val_score

sklearn.metrics.r2_score
```