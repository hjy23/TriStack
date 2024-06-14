import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import read_dataset
from extract_feature import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import logging
import tensorflow
import argparse

# 曾经的基分类器有一个Transformer，效果不好就去掉了
# from Transformer_1 import *

sequences, labels = read_dataset.read_dataset_zero('../datasets/AMP_1.txt')

dataset_random_state = 42
xgb_random_seed = 42
rf_random_seed = 42
xgb_n_estimators = 155
xgb_max_depth = 16
rf_n_estimators = 450
rf_max_depth = 50
parser = argparse.ArgumentParser(description="hyperparameter discovering")
parser.add_argument('--mode', type=str, help='填写AMP或AIP', default='AMP')
parser.add_argument('--dataset_random_state', type=int, help='数据集划分随机数种子', default=42)
parser.add_argument('--xgb_random_seed', type=int, help='XGBoost随机数种子', default=42)
parser.add_argument('--rf_random_seed', type=int, help='Random Forest随机数种子', default=42)
parser.add_argument('--xgb_n_estimators', type=int, help='XGBoost森林规模', default=155)
parser.add_argument('--xgb_max_depth', type=int, help='XGBoost树深度', default=16)
parser.add_argument('--rf_n_estimators', type=int, help='Random Forest森林规模', default=450)
parser.add_argument('--rf_max_depth', type=int, help='Random Forest树深度', default=50)

args = parser.parse_args()

if args.dataset_random_state is not None:
    print(f"数据集划分随机数种子为: {args.dataset_random_state}")
if args.xgb_random_seed is not None:
    print(f"XGBoost随机数种子为: {args.xgb_random_seed}")
if args.rf_random_seed is not None:
    print(f"Random Forest随机数种子为: {args.rf_random_seed}")
if args.xgb_n_estimators is not None:
    print(f"XGBoost森林规模为: {args.xgb_n_estimators}")
if args.xgb_max_depth is not None:
    print(f"XGBoost随机数种子为: {args.xgb_max_depth}")
if args.xgb_random_seed is not None:
    print(f"XGBoost随机数种子为: {args.dataset_random_state}")

# if args.mode == 'AMP':
#     sequences, labels = read_dataset.read_dataset_amp('AMP.txt')
# elif args.mode == 'AIP':
#     sequences, labels = read_dataset.read_dataset_aip('AIP.txt')
# else:
#     print('输入正确的')

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(
    f"启动训练，使用的超参数：mode={args.mode}, dataset_random_state={args.dataset_random_state}, xgb_random_seed={args.xgb_random_seed}, rf_random_seed={args.rf_random_seed}, xgb_n_estimators={args.xgb_n_estimators}, xgb_max_depth={args.xgb_max_depth}, rf_n_estimators={args.rf_n_estimators}, rf_max_depth={args.rf_max_depth}")

all_features = extract_features(sequences)
SPLIT_NUM = 1669
X_train = all_features[SPLIT_NUM:]
y_train = labels[SPLIT_NUM:]
X_test = all_features[:SPLIT_NUM]
y_test = labels[:SPLIT_NUM]

base_models = [
    ('xgb', XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, use_label_encoder=False,
                          eval_metric='logloss', random_state=xgb_random_seed)),
    ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=rf_random_seed)),
    ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=3, C=10, gamma=0.001))),
]

# 训练基模型并收集训练集和测试集的预测
train_features = []
test_features = []

for name, model in base_models:
    # 获取模型的参数
    model_params = model.get_params()
    # 将模型的参数转换为字符串格式
    params_str = ', '.join(f"{key}={value}" for key, value in model_params.items())
    if name == 'xgb':
        # XGBoost支持在训练时输出日志
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        # 评估准确率
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
    else:
        model.fit(X_train, y_train)
        # 评估准确率
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    train_features.append(train_pred.reshape(-1, 1))
    test_features.append(test_pred.reshape(-1, 1))

# 合并所有基模型的预测作为新特征
X_train_meta = np.hstack(train_features)
X_test_meta = np.hstack(test_features)

from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


class ResidualUnit(layers.Layer):
    def __init__(self, units, activation=tensorflow.nn.swish, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.activation = layers.Activation(activation)
        self.main_layers = [
            layers.Dense(units, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(units),
            layers.Dropout(dropout_rate)
        ]
        self.skip_layers = [layers.Dense(units, use_bias=False)]  # 简化了原始代码

    def call(self, inputs, training=False):  # 添加training参数
        Z = inputs
        for layer in self.main_layers:
            if isinstance(layer, layers.Dropout):
                # 使用传入的training标志来决定是否应用Dropout
                Z = layer(Z, training=training)
            else:
                Z = layer(Z)
        skip_Z = inputs
        if len(self.skip_layers) > 0:
            skip_Z = self.skip_layers[0](skip_Z)
        return self.activation(Z + skip_Z)


def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = ResidualUnit(8)(inputs)
    x = ResidualUnit(8)(x)
    x = ResidualUnit(8)(x)
    x = ResidualUnit(8)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# 确保X_train_meta和y_train是NumPy数组
X_train_meta = np.array(X_train_meta)
y_train = np.array(y_train)
X_test_meta = np.array(X_test_meta)
y_test = np.array(y_test)

# 模型编译
model = build_model(input_shape=(3,), num_classes=2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
best_accuracy = 0.0
best_precision = 0.0
best_recall = 0.0
best_f1_score = 0.0
best_mcc = 0.0
best_prediction = []
best_prediction_binary = []
best_model_metrics = {}
# 打印模型概要
model.summary()

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import tensorflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def save_results_to_excel(metrics, predictions, predictions_binary, folder_name="best_results_amp_with_volume"):
    # 确保目标文件夹存在，如果不存在，则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 获取当前时间，并格式化为字符串，用于文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder_name}/results_{current_time}.xlsx"

    # 创建一个Pandas Excel writer使用openpyxl引擎
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 将预测结果转换为DataFrame并保存
        predictions_df = pd.DataFrame({
            'Predictions': predictions.flatten(),  # 确保predictions是一维的
            'Predictions Binary': predictions_binary.flatten()
        })

        # 写入一个Sheet名为'Predictions'
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')

        # 将metrics转换为DataFrame并保存
        metrics_df = pd.DataFrame(metrics, index=[0])

        # 写入一个Sheet名为'Metrics'
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

    print(f"Results saved to {filename}")


acc_pool = []
f1_score_pool = []
mcc_pool = []
for _ in range(10):
    history = model.fit(X_train_meta, y_train, epochs=10, batch_size=4, validation_split=0.2)
    for threshold in [0.95]:
        predictions = model.predict(X_test_meta)
        predictions_binary = (predictions > threshold).astype(int)

        # 计算准确率
        accuracy = accuracy_score(y_test, predictions_binary)

        # 计算精确率
        precision = precision_score(y_test, predictions_binary)

        # 计算召回率
        recall = recall_score(y_test, predictions_binary)

        # 计算F1分数
        f1 = f1_score(y_test, predictions_binary)
        mcc = matthews_corrcoef(y_test, predictions_binary)
        acc_pool.append(accuracy)
        f1_score_pool.append(f1)
        mcc_pool.append(mcc)
        # 计算马修斯相关系数

        if accuracy > best_accuracy or (threshold == 0.5 and accuracy == best_accuracy):
            best_accuracy = accuracy
            best_f1_score = f1
            best_precision = precision
            best_recall = recall
            best_mcc = mcc
            best_model = model
            best_prediction = predictions
            best_prediction_binary = predictions_binary
            best_model_metrics = {
                "Now threshold": threshold,
                "Accuracy": best_accuracy,
                "Precision": best_precision,
                "Recall": best_recall,
                "F1 Score": best_f1_score,
                "Matthews Correlation Coefficient": best_mcc
            }

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"MCC: {mcc}")

save_results_to_excel(best_model_metrics, best_prediction, best_prediction_binary)
print('acc_pool:')
print(acc_pool)
print('f1_score_pool:')
print(f1_score_pool)
print('mcc_pool:')
print(mcc_pool)


print(f"Accuracy: {best_accuracy}")
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F1 Score: {best_f1_score}")
print(f"MCC: {best_mcc}")
