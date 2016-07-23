import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


cls_data = pd.read_csv('classification.csv', header=0).as_matrix()

TP = len(cls_data[(cls_data[:, 0] == 1) & (cls_data[:, 1] == 1)])
FP = len(cls_data[(cls_data[:, 0] == 0) & (cls_data[:, 1] == 1)])
TN = len(cls_data[(cls_data[:, 0] == 0) & (cls_data[:, 1] == 0)])
FN = len(cls_data[(cls_data[:, 0] == 1) & (cls_data[:, 1] == 0)])

print("%d %d %d %d" % (TP, FP, FN, TN))

accuracy = accuracy_score(cls_data[:, 0], cls_data[:, 1])
precision = precision_score(cls_data[:, 0], cls_data[:, 1])
recall = recall_score(cls_data[:, 0], cls_data[:, 1])
f1 = f1_score(cls_data[:, 0], cls_data[:, 1])

print("%.3f %.3f %.3f %.3f" % (accuracy, precision, recall, f1))

scores = pd.read_csv('scores.csv', header=0).as_matrix()

lr_auc = roc_auc_score(scores[:, 0], scores[:, 1])
svm_auc = roc_auc_score(scores[:, 0], scores[:, 2])
knn_auc = roc_auc_score(scores[:, 0], scores[:, 3])
tree_auc = roc_auc_score(scores[:, 0], scores[:, 4])

print("rocauc logreg %.3f, svm %.3f, knn %.3f, tree %.3f" % (lr_auc, svm_auc, knn_auc, tree_auc))

lr_precision, lr_recall, _ = precision_recall_curve(scores[:, 0], scores[:, 1])
svm_precision, svm_recall, _ = precision_recall_curve(scores[:, 0], scores[:, 2])
knn_precision, knn_recall, _ = precision_recall_curve(scores[:, 0], scores[:, 3])
tree_precision, tree_recall, _ = precision_recall_curve(scores[:, 0], scores[:, 4])

lr_answer = max(lr_precision[lr_recall >= 0.7])
svm_answer = max(svm_precision[svm_recall >= 0.7])
knn_answer = max(knn_precision[knn_recall >= 0.7])
tree_answer = max(tree_precision[tree_recall >= 0.7])

print("precision logreg %.3f, svm %.3f, knn %.3f, tree %.3f" % (lr_answer, svm_answer, knn_answer, tree_answer))
