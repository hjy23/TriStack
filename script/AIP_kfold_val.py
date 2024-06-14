import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from extract_feature import extract_features
import read_dataset
from dde_cksaap import *
from properties import *

xgb_random_seed = 42
rf_random_seed = 42
xgb_n_estimators = 33
xgb_max_depth = 10
rf_n_estimators = 325
rf_max_depth = 50


# 假设 read_dataset 和 extract_features 是已经定义好的函数
sequences, labels = read_dataset.read_dataset_from_aipstack_work('AIP_6.txt')
sequences, labels = read_dataset.read_dataset_zero('AMP_1.txt')

# 先划分数据集
# X_train_seq, X_test_seq, y_train, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=dataset_random_state)
# X_train_seq = sequences[838:]
# X_val_seq = sequences[419:838]
# X_test_seq = sequences[:419]
# y_train = labels[838:]
# y_val = labels[419:838]
# y_test = labels[:419]
X_train_seq = sequences[3338:]
X_val_seq = sequences[1669:3338]
X_test_seq = sequences[:1669]
y_train = labels[3338:]
y_val = labels[1669:3338]
y_test = labels[:1669]

# 再对训练集和测试集序列分别提取特征
X_train = extract_features(X_train_seq)
X_test = extract_features(X_test_seq)

base_models = [
    ('xgb',
     XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, use_label_encoder=False,
                   eval_metric='logloss', random_state=xgb_random_seed)),
    ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=rf_random_seed)),
    ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=6))),
]
# 3

# 训练基模型并收集训练集和测试集的预测
train_features = []
test_features = []

# 假设 extract_features, sequences, labels, xgb_n_estimators, xgb_max_depth, xgb_random_seed, rf_n_estimators, rf_max_depth, rf_random_seed 已经定义

# 首先，你需要合并训练集和验证集（如果你想在交叉验证中使用它们）
X_combined_seq = np.concatenate((X_train_seq, X_val_seq))
y_combined = np.concatenate((y_train, y_val))

# 对合并后的数据集提取特征
X_combined = extract_features(X_combined_seq)

# 假设 X_combined, y_combined 已经定义，这里使用全数据集作为例子

# 参数范围
n_estimators_range = [50, 100, 200, 300, 500, 800]
max_depth_range = [6, 10, 15, 30, 50, 80]

# 初始化记录性能的数组
scores = np.zeros((len(n_estimators_range), len(max_depth_range)))
#
# # 定义5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# 定义参数网格
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.001, 0.01, 0.1, 1]
}

# 创建SVC模型的管道，包括标准化处理
pipe = make_pipeline(StandardScaler(), SVC(probability=True, random_state=6))

# 定义5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建GridSearchCV对象
grid_search = GridSearchCV(pipe, param_grid, cv=kf, scoring='accuracy', verbose=3)

# 执行网格搜索
grid_search.fit(X_combined, y_combined)

# 打印最佳参数组合和对应的准确率
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

# 获取最佳模型
best_model = grid_search.best_estimator_

import matplotlib.pyplot as plt
import numpy as np

# 获取GridSearchCV的结果
cv_results = grid_search.cv_results_

# 提取网格搜索的参数和对应的平均测试得分
C_range = grid_search.param_grid['svc__C']
gamma_range = grid_search.param_grid['svc__gamma']
scores = cv_results['mean_test_score'].reshape(len(C_range), len(gamma_range))

# 绘制热力图
plt.figure(figsize=(8, 6))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Grid Search Accuracy Score')
plt.show()

# 打印最佳参数和最佳得分
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

# # 循环不同的n_estimators和max_depth值
# for i, n_estimators in enumerate(n_estimators_range):
#     for j, max_depth in enumerate(max_depth_range):
#         # 定义模型
#         # model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, use_label_encoder=False,
#         #                       eval_metric='logloss', random_state=42)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
#         # 计算当前参数组合下的交叉验证得分
#         cv_scores = cross_val_score(model, X_combined, y_combined, cv=kf, scoring='accuracy')
#
#         # 记录平均得分
#         scores[i, j] = cv_scores.mean()
#
#         # 输出当前参数组合下的交叉验证平均得分
#         print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, CV Accuracy: {scores[i, j]:.4f}")
#
# # 找到得分最高的参数组合
# max_score_idx = np.unravel_index(scores.argmax(), scores.shape)
# best_n_estimators = n_estimators_range[max_score_idx[0]]
# best_max_depth = max_depth_range[max_score_idx[1]]
#
# print(f"\nBest n_estimators: {best_n_estimators}, Best max_depth: {best_max_depth}")
#
# # 绘制结果
# fig, ax = plt.subplots()
# cax = ax.matshow(scores, cmap='viridis')
# fig.colorbar(cax)
#
# ax.set_xticklabels([''] + max_depth_range)
# ax.set_yticklabels([''] + n_estimators_range)
#
# ax.set_xlabel('Max Depth')
# ax.set_ylabel('N Estimators')
#
# for (i, j), val in np.ndenumerate(scores):
#     ax.text(j, i, f"{val:.3f}", va='center', ha='center')
#
# plt.show()
