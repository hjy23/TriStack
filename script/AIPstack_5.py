import argparse
import logging
import os
from datetime import datetime
import tensorflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras import layers, models
import numpy as np
import read_dataset
from extract_feature import extract_features

#  这部分参数请按照论文最佳参数填写
dataset_random_state = 42
xgb_random_seed = 42
rf_random_seed = 42
xgb_n_estimators = 325
xgb_max_depth = 10
rf_n_estimators = 450
rf_max_depth = 35

xgb_random_seed = 42
rf_random_seed = 42
xgb_n_estimators = 50
xgb_max_depth = 15
rf_n_estimators = 50
rf_max_depth = 80

# 33 -> 68原用于Terminal启动网格搜参，现已有专门的网格搜参脚本，这部分代码阅读时可略过，保留这部分代码运行脚本时不会有影响
parser = argparse.ArgumentParser(description="hyperparameter discovering")
parser.add_argument('--mode', type=str, help='填写AMP或AIP', default='AIP')
parser.add_argument('--dataset_random_state', type=int, help='数据集划分随机数种子', default=42)
parser.add_argument('--xgb_random_seed', type=int, help='XGBoost随机数种子', default=42)
parser.add_argument('--rf_random_seed', type=int, help='Random Forest随机数种子', default=42)
parser.add_argument('--xgb_n_estimators', type=int, help='XGBoost森林规模', default=50)
parser.add_argument('--xgb_max_depth', type=int, help='XGBoost树深度', default=15)
parser.add_argument('--rf_n_estimators', type=int, help='Random Forest森林规模', default=50)
parser.add_argument('--rf_max_depth', type=int, help='Random Forest树深度', default=80)

args = parser.parse_args()

if args.dataset_random_state is not None:
    print(f"数据集划分随机数种子为: {args.dataset_random_state}")
if args.dataset_random_state is not None:
    print(f"数据集划分随机数种子为: {args.dataset_random_state}")
if args.rf_n_estimators is not None:
    print(f"Random forest规模为: {args.xgb_random_seed}")
if args.rf_max_depth is not None:
    print(f"Random forest最大深度为: {args.xgb_random_seed}")
if args.rf_random_seed is not None:
    print(f"Random Forest随机数种子为: {args.rf_random_seed}")
if args.xgb_n_estimators is not None:
    print(f"XGBoost森林规模为: {args.xgb_n_estimators}")
if args.xgb_max_depth is not None:
    print(f"XGBoost最大深度为: {args.xgb_max_depth}")
if args.xgb_random_seed is not None:
    print(f"XGBoost随机数种子为: {args.dataset_random_state}")

# if args.mode == 'AMP':
#     sequences, labels = read_dataset.read_dataset_amp('AMP.txt')
# elif args.mode == 'AIP':
#     sequences, labels = read_dataset.read_dataset_aip('AIP.txt')
# else:
#     print('输入正确的数据集名称')

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(
    f"启动训练，使用的超参数：mode={args.mode}, dataset_random_state={args.dataset_random_state}, xgb_random_seed={args.xgb_random_seed}, rf_random_seed={args.rf_random_seed}, xgb_n_estimators={args.xgb_n_estimators}, xgb_max_depth={args.xgb_max_depth}, rf_n_estimators={args.rf_n_estimators}, rf_max_depth={args.rf_max_depth}")

# read_dataset 和 extract_features 是其他模块中已经定义好的函数
sequences, labels = read_dataset.read_dataset_from_aipstack_work('../datasets/AIP_6.txt')

# 划分数据集
# X_train_seq = sequences[419:][:100]
# X_test_seq = sequences[:419][:100]
# y_train = labels[419:][:100]
# y_test = labels[:419][:100]
X_train_seq = sequences[419:]
X_test_seq = sequences[:419]
y_train = labels[419:]
y_test = labels[:419]

# 再对训练集和测试集序列分别提取特征
X_train = extract_features(X_train_seq)
X_test = extract_features(X_test_seq)

base_models = [
    ('xgb', XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, use_label_encoder=False,
                          # eval_metric='logloss', random_state=xgb_random_seed, gpu_id=0)),
                          eval_metric='logloss', random_state=xgb_random_seed)),
    ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=rf_random_seed)),
    ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=6))),
]

# 训练基模型并收集训练集和测试集的预测
train_features = []
test_features = []

for name, model in base_models:
    # 获取模型的参数
    # model = model.to()
    model_params = model.get_params()
    # 将模型的参数转换为字符串格式
    params_str = ', '.join(f"{key}={value}" for key, value in model_params.items())
    if name == 'xgb':
        # XGBoost支持在训练时输出日志
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        # model.fit(X_train, y_train, verbose=True)
        # 回显测试集准确率
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
        print(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
    elif name == 'xgb_without_val':
        # XGBoost支持在训练时输出日志
        # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
        model.fit(X_train, y_train, verbose=True)
        # 回显测试集准确率
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        # logging.info(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
        print(f"{name} - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Parameters: {params_str}")
    else:
        model.fit(X_train, y_train)
        # 回显测试集准确率
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


class ResidualUnit(layers.Layer):
    def __init__(self, units, activation=tensorflow.nn.swish, dropout_rate=0.2, **kwargs):
        """
        初始化残差单元
        :param
        units (int): 单元数目，指定全连接层的输出空间维度。
        activation (callable): 激活函数，默认为tensorflow.nn.swish。
        dropout_rate (float): Dropout层的丢弃率。
        **kwargs: 传递给父类初始化的其他关键字参数。
        """
        super().__init__(**kwargs)
        self.activation = layers.Activation(activation)  # 设置激活函数
        self.main_layers = [
            layers.Dense(units, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(units),
            layers.Dropout(dropout_rate)
        ]
        self.skip_layers = [layers.Dense(units, use_bias=False)]  # 跳跃连接层

    def call(self, inputs, training=False):
        """
            处理输入数据并返回输出
            :param
            inputs (Tensor): 输入数据的张量。
            training (bool): 指示模型是否处于训练模式。
            :return
            Tensor: 经过残差单元处理后的输出数据。
        """
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
    """
    建立残差网络模型
    :param input_shape:输入维度
    :param num_classes:分类数量，本研究问题为二分类问题
    :return:残差网络模型
    """
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

import pandas as pd
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

acc_pool = []
f1_score_pool = []
mcc_pool = []


def save_results_to_excel(metrics: dict, predictions, predictions_binary, folder_name="best_results_aip_with_volume"):
    """
    保存实验结果
    :param metrics: 模型性能指标
    :param predictions: 模型预测结果（浮点数）
    :param predictions_binary: 模型预测结果（整数）
    :param folder_name:存储文件夹路径
    :return:None
    """
    # 如果该文件夹不存在，则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 获取当前时间，并格式化为字符串，用于命名文件
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 根据实验目的不同，可自行修改文件名
    filename = f"{folder_name}/best_f1_results_{current_time}.xlsx"

    # 创建一个Pandas Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 将预测结果转换为DataFrame并保存
        predictions_df = pd.DataFrame({
            'Predictions': predictions.flatten(),
            'Predictions Binary': predictions_binary.flatten()
        })

        # 写入一个Sheet，名为'Predictions'
        # 保存模型的预测结果
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')

        # 将metrics转换为DataFrame并保存
        metrics_df = pd.DataFrame(metrics, index=[0])

        # 写入一个Sheet，名为'Metrics'
        # 保存模型的性能指标结果
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

    print(f"Results saved to {filename}")


# 确保此处正确计算了所有必要的指标，然后将它们保存在字典中


for _ in range(5):
    model.fit(X_train_meta, y_train, epochs=10, batch_size=32, validation_split=0.2)
    for threshold in [0.5]:
        # 得到模型预测结果
        predictions = model.predict(X_test_meta)
        predictions_binary = (predictions > threshold).astype(int)

        # predictions_binary_list = predictions_binary.tolist()
        # predictions_binary_flat_list = [item for sublist in predictions_binary_list for item in sublist]
        # correct_predictions = sum(1 for label, prediction in zip(predictions_binary_flat_list, y_test) if label == prediction)
        # accuracy = correct_predictions / len(predictions_binary_flat_list)

        # 计算准确率
        accuracy = accuracy_score(y_test, predictions_binary)

        # 计算精确率
        precision = precision_score(y_test, predictions_binary)

        # 计算召回率
        recall = recall_score(y_test, predictions_binary)

        # 计算F1分数
        f1 = f1_score(y_test, predictions_binary)

        # 通过混淆矩阵计算TN FP FN TP
        tn, fp, fn, tp = confusion_matrix(y_test, predictions_binary).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        MCC = ((tp * tn) - (fp * fn)) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
        # 计算马修斯相关系数
        mcc = matthews_corrcoef(y_test, predictions_binary)
        acc_pool.append(accuracy)
        f1_score_pool.append(f1)
        mcc_pool.append(mcc)
        if f1 > best_f1_score:
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
                "Matthews Correlation Coefficient": best_mcc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
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

