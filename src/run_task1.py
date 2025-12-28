#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reproducibility script (exported from notebook).

Place dataset files under data/raw/ before running.
"""

import pandas as pd

def process_data(file_name):
    # 提取批次名称，去掉路径和扩展名 (CN comment)
    batch_name = file_name.split('/')[-1].replace('.dat', '') + '_data'

    # 读取数据文件 (CN comment)
    data = pd.read_csv(file_name, sep=' ', header=None)
    # 创建一个新的DataFrame来存储处理后的数据 (CN comment)
    new_data = pd.DataFrame()

    # 遍历每一行 (CN comment)
    for index, row in data.iterrows():
        # 提取第一列的值作为气体标号 (CN comment)
        gas_label = row[0]
        # 创建一个临时的字典来存储键值对 (CN comment)
        temp_dict = {"gas_label": gas_label}  # 首先保存气体标号
        # 遍历其他列，处理键值对 (CN comment)
        for item in row[1:]:
            if ':' in str(item):
                key, value = str(item).split(':')  # 分割键和值
                temp_dict[int(key)] = float(value)  # 将键和值加入临时字典
        # 将临时字典转换为DataFrame，并追加到新的DataFrame (CN comment)
        new_row = pd.DataFrame([temp_dict])
        new_data = pd.concat([new_data, new_row], ignore_index=True)

    # 重置索引 (CN comment)
    new_data.reset_index(drop=True, inplace=True)

    return new_data

import pandas as pd
from tensorflow.keras.utils import to_categorical

def prepare_data(data):
    # 提取标签并将其从1-6转换为0-5 (CN comment)
    Y = data['gas_label'].values - 1
    
    # 提取特征 (CN comment)
    X = data.drop('gas_label', axis=1).values
    
    # 将标签转换为独热编码形式 (CN comment)
    Y_encoded = to_categorical(Y, num_classes=6)
    
    return X, Y_encoded

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# 固定随机种子 (CN comment)
np.random.seed(1)
tf.random.set_seed(1)

# 定义模型创建函数 (CN comment)
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(6, activation='softmax'))  # 6个分类
    return model

#baseline

#baseline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

results = []
detailed_accuracies = []

# 使用批次1作为训练数据 (CN comment)
X_train, Y_train = all_data[0]

for j in range(2, 11):
    Xt, Yt = all_data[j-1]  # 当前批次用作测试

    # 进行多次实验 (CN comment)
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []
    
    test_accuracies = []
    test_f1_scores = []
    test_precisions = []
    test_recalls = []
    
    for experiment in range(1, 31):
        # Split the target batch into validation and test sets
        X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

        # 创建并训练模型 (CN comment)
        model = create_model(X_train.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        
        # Evaluate on the validation set
        val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
        val_accuracies.append(val_accuracy)

        Y_val_pred = model.predict(X_val)
        Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
        Y_val_classes = np.argmax(Y_val, axis=1)
        val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        
        # Evaluate on the test set
        test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
        test_accuracies.append(test_accuracy)

        Y_test_pred = model.predict(X_test)
        Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
        Y_test_classes = np.argmax(Y_test, axis=1)
        test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        
        test_f1_scores.append(test_f1)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        
        detailed_accuracies.append({
            'Model': f'model_{j}', 
            'Experiment': experiment, 
            'Val Accuracy': val_accuracy, 
            'Val F1 Score': val_f1, 
            'Val Precision': val_precision, 
            'Val Recall': val_recall, 
            'Test Accuracy': test_accuracy, 
            'Test F1 Score': test_f1, 
            'Test Precision': test_precision, 
            'Test Recall': test_recall
        })
    
    # 计算验证集和测试集的平均值和标准差 (CN comment)
    avg_val_accuracy = np.mean(val_accuracies)
    avg_val_f1 = np.mean(val_f1_scores)
    avg_val_precision = np.mean(val_precisions)
    avg_val_recall = np.mean(val_recalls)
    
    avg_test_accuracy = np.mean(test_accuracies)
    avg_test_f1 = np.mean(test_f1_scores)
    avg_test_precision = np.mean(test_precisions)
    avg_test_recall = np.mean(test_recalls)
    
    std_dev_val = np.std(val_accuracies)
    std_dev_test = np.std(test_accuracies)
    
    results.append({
        'Model': f'model_{j}', 
        'Average Val Accuracy': avg_val_accuracy, 
        'Val Standard Deviation': std_dev_val, 
        'Average Val F1 Score': avg_val_f1, 
        'Average Val Precision': avg_val_precision, 
        'Average Val Recall': avg_val_recall,
        'Average Test Accuracy': avg_test_accuracy, 
        'Test Standard Deviation': std_dev_test, 
        'Average Test F1 Score': avg_test_f1, 
        'Average Test Precision': avg_test_precision, 
        'Average Test Recall': avg_test_recall
    })

# Convert results to DataFrame
detailed_results_df = pd.DataFrame(detailed_accuracies)
summary_results_df = pd.DataFrame(results)

# Save detailed results to an Excel file
detailed_save_path = "outputs/一对一详细.xlsx"
detailed_results_df.to_excel(detailed_save_path, index=False)

# 保存汇总结果到Excel文件 (CN comment)
summary_save_path = "outputs/一对一汇总.xlsx"
summary_results_df.to_excel(summary_save_path, index=False)

# Print summary results进行检查 (CN comment)
print(summary_results_df)

#KD

#使用知识蒸馏完成 (CN comment)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
# 输入特征维度 (CN comment)
input_dim = 128  # 假设输入特征数为128

# 生成模型 (CN comment)
teacher_model = create_model(input_dim)
student_model = create_model(input_dim)
# Distillation loss function
def distillation_loss(y_true, y_pred, temperature=100):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# 编译教师模型 (CN comment)
teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 编译学生模型 (CN comment)
student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred), metrics=['accuracy'])

# KD 7.28 再次确认一下温度 (CN comment)
# 用知识蒸馏给每个任务定制温度 (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# Define candidate temperature values
temperature_options = [0.3, 1, 2, 3, 5, 25, 50, 100, 200]

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

# Initialize result containers
results = []
detailed_accuracies = []

# 使用batch 1作为训练集，其余批次分别作为测试集 (CN comment)
Xs, Ys = all_data[0]  # batch 1的数据作为源域数据

for j in range(2, 11):
    Xt, Yt = all_data[j-1]  # 当前批次用作目标域数据

    # 寻找最佳温度 (CN comment)
    best_accuracy = 0
    best_temperature = None
    best_val_metrics = []
    best_test_metrics = []

    for temperature in temperature_options:
        val_accuracies = []
        val_f1_scores = []
        val_precisions = []
        val_recalls = []
        test_accuracies = []
        test_f1_scores = []
        test_precisions = []
        test_recalls = []

        for experiment in range(1, 11):  # 进行10次实验
            # Split the target batch into validation and test sets
            X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

            # Initialize teacher/student models
            teacher_model = create_model(Xs.shape[1])
            student_model = create_model(Xs.shape[1])
            teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, temperature), metrics=['accuracy'])

            # Train the teacher model
            teacher_model.fit(Xs, Ys, epochs=20, batch_size=32, verbose=0)

            # Generate soft labels for training and validation sets
            soft_labels_train = teacher_model.predict(Xs)
            soft_labels_val = teacher_model.predict(X_val)

            # Merge soft labels
            soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
            X_combined_with_val = np.vstack((Xs, X_val))

            # Train the student model
            student_model.fit(X_combined_with_val, soft_labels_combined, epochs=20, batch_size=32, verbose=0)

            # Evaluate on the validation set
            val_loss, val_accuracy = student_model.evaluate(X_val, Y_val, verbose=0)
            val_accuracies.append(val_accuracy)

            Y_val_pred = student_model.predict(X_val)
            Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
            Y_val_classes = np.argmax(Y_val, axis=1)
            val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
            val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
            val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

            val_f1_scores.append(val_f1)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)

            # Evaluate on the test set
            test_loss, test_accuracy = student_model.evaluate(X_test, Y_test, verbose=0)
            test_accuracies.append(test_accuracy)

            Y_test_pred = student_model.predict(X_test)
            Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
            Y_test_classes = np.argmax(Y_test, axis=1)
            test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
            test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
            test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

            test_f1_scores.append(test_f1)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)

            detailed_accuracies.append({'Model': f'student_model_{j}', 'Experiment': f'Temperature={temperature}', 'Val Accuracy': val_accuracy, 'Val F1 Score': val_f1, 'Val Precision': val_precision, 'Val Recall': val_recall, 'Test Accuracy': test_accuracy, 'Test F1 Score': test_f1, 'Test Precision': test_precision, 'Test Recall': test_recall, 'Repeat': experiment})

        avg_val_accuracy = np.mean(val_accuracies)
        avg_val_f1 = np.mean(val_f1_scores)
        avg_val_precision = np.mean(val_precisions)
        avg_val_recall = np.mean(val_recalls)
        avg_test_accuracy = np.mean(test_accuracies)
        avg_test_f1 = np.mean(test_f1_scores)
        avg_test_precision = np.mean(test_precisions)
        avg_test_recall = np.mean(test_recalls)

        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            best_temperature = temperature
            best_val_metrics = [{'Val Accuracy': a, 'Val F1 Score': f, 'Val Precision': p, 'Val Recall': r} for a, f, p, r in zip(val_accuracies, val_f1_scores, val_precisions, val_recalls)]
            best_test_metrics = [{'Test Accuracy': a, 'Test F1 Score': f, 'Test Precision': p, 'Test Recall': r} for a, f, p, r in zip(test_accuracies, test_f1_scores, test_precisions, test_recalls)]

    # Record the best result
    summary_results = {
        'Model': f'student_model_{j}',
        'Best Temperature': best_temperature,
        'Val Accuracy': [m['Val Accuracy'] for m in best_val_metrics],
        'Val F1 Score': [m['Val F1 Score'] for m in best_val_metrics],
        'Val Precision': [m['Val Precision'] for m in best_val_metrics],
        'Val Recall': [m['Val Recall'] for m in best_val_metrics],
        'Test Accuracy': [m['Test Accuracy'] for m in best_test_metrics],
        'Test F1 Score': [m['Test F1 Score'] for m in best_test_metrics],
        'Test Precision': [m['Test Precision'] for m in best_test_metrics],
        'Test Recall': [m['Test Recall'] for m in best_test_metrics],
        'Avg Val Accuracy': best_accuracy,
        'Avg Val F1 Score': avg_val_f1,
        'Avg Val Precision': avg_val_precision,
        'Avg Val Recall': avg_val_recall,
        'Avg Test Accuracy': avg_test_accuracy,
        'Avg Test F1 Score': avg_test_f1,
        'Avg Test Precision': avg_test_precision,
        'Avg Test Recall': avg_test_recall
    }

    results.append(summary_results)

# Convert results to DataFrame
detailed_results_df = pd.DataFrame(detailed_accuracies)
summary_results_df = pd.DataFrame(results)

# Save results to Excel files
detailed_results_df.to_excel("outputs/知识蒸馏一对一详细test.xlsx", index=False)
summary_results_df.to_excel("outputs/知识蒸馏一对一汇总test.xlsx", index=False)

# Print summary results
print(summary_results_df)

#30遍 (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

# 从汇总文件读取最佳温度并进行30次重复实验 (CN comment)
for j in range(2, 11):
    summary_path = "outputs/知识蒸馏一对一汇总test.xlsx"
    summary_df = pd.read_excel(summary_path)

    best_temperature = summary_df.loc[summary_df['Model'] == f'student_model_{j}', 'Best Temperature'].values[0]

    results = []
    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs, Ys = all_data[0]  # batch 1的数据作为源域数据

    for experiment in range(1, 31):  # 进行30次实验
        X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

        # Initialize teacher/student models
        teacher_model = create_model(Xs.shape[1])
        student_model = create_model(Xs.shape[1])
        teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, best_temperature), metrics=['accuracy'])

        # Train the teacher model
        teacher_model.fit(Xs, Ys, epochs=20, batch_size=32, verbose=0)

        # Generate soft labels for training and validation sets
        soft_labels_train = teacher_model.predict(Xs)
        soft_labels_val = teacher_model.predict(X_val)

        # Merge soft labels
        soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
        X_combined_with_val = np.vstack((Xs, X_val))

        # Train the student model
        student_model.fit(X_combined_with_val, soft_labels_combined, epochs=20, batch_size=32, verbose=0)

        # Evaluate on the validation set
        val_loss, val_accuracy = student_model.evaluate(X_val, Y_val, verbose=0)
        Y_val_pred = student_model.predict(X_val)
        Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
        Y_val_classes = np.argmax(Y_val, axis=1)
        val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

        # Evaluate on the test set
        test_loss, test_accuracy = student_model.evaluate(X_test, Y_test, verbose=0)
        Y_test_pred = student_model.predict(X_test)
        Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
        Y_test_classes = np.argmax(Y_test, axis=1)
        test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

        detailed_accuracies.append({
            'Model': f'student_model_{j}', 
            'Experiment': experiment, 
            'Temperature': best_temperature, 
            'Val Accuracy': val_accuracy, 
            'Val F1 Score': val_f1, 
            'Val Precision': val_precision, 
            'Val Recall': val_recall, 
            'Test Accuracy': test_accuracy, 
            'Test F1 Score': test_f1, 
            'Test Precision': test_precision, 
            'Test Recall': test_recall
        })

    # Convert results to DataFrame
    detailed_results_df = pd.DataFrame(detailed_accuracies)

    # Save detailed results to an Excel file
    detailed_save_path = f"outputs/知识蒸馏一对一详细_batch_{j}_30遍.xlsx"
    detailed_results_df.to_excel(detailed_save_path, index=False)

    # 打印完成信息 (CN comment)
    print(f"Batch {j} - 30 repetitions completed and saved to {detailed_save_path}")

#DRCA

#定义DRCA (CN comment)
import numpy as np

class DRCA():
    '''
    The DRCA Class
    '''

    def __init__(self, n_components=2, alpha=None):
        self.Sw_s = None
        self.Sw_t = None
        self.mu_s = None
        self.mu_t = None
        self.alpha = alpha
        self.D_tilde = n_components

    def fit(self, Xs, Xt):
        Ns = Xs.shape[0]
        Nt = Xt.shape[0]
        D = Xs.shape[1]
        self.mu_s = np.mean(Xs, axis=0, keepdims=True)
        self.mu_t = np.mean(Xt, axis=0, keepdims=True)
        self.Sw_s = (Xs - self.mu_s).T @ (Xs - self.mu_s)
        self.Sw_t = (Xt - self.mu_t).T @ (Xt - self.mu_t)
        if self.alpha is None:
            self.alpha = Ns / Nt
        self.nominator = self.Sw_s + self.Sw_t * self.alpha
        self.denominator = (self.mu_s - self.mu_t).T @ (self.mu_s - self.mu_t)
        eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(self.denominator) @ self.nominator)
        idx = np.abs(eigenValues).argsort()[::-1]
        self.eigenValues = eigenValues[idx]
        self.eigenVectors = eigenVectors[:, idx]
        self.W = self.eigenVectors[:, 0:self.D_tilde]

    def transform(self, X):
        return np.real(np.matmul(X, self.W))

    def fit_transform(self, Xs, Xt):
        self.fit(Xs, Xt)
        return self.transform(Xs), self.transform(Xt)

# DRCA 7.30
# 定制DRCA (CN comment)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

# 定义DRCA的可选参数 (CN comment)
n_components_options = [50, 100, 150, 200, 300, 500]
alpha_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Initialize result containers
results = []
detailed_accuracies = []

# 使用batch 1作为训练集，其余批次分别作为测试集 (CN comment)
Xs, Ys = all_data[0]  # batch 1的数据作为源域数据

for j in range(2, 11):
    Xt, Yt = all_data[j-1]  # 当前批次用作目标域数据

    best_accuracy = 0
    best_n_components = None
    best_alpha = None
    best_val_metrics = []
    best_test_metrics = []

    # 遍历所有n_components和alpha组合 (CN comment)
    for n_components in n_components_options:
        for alpha in alpha_options:
            val_accuracies = []
            val_f1_scores = []
            val_precisions = []
            val_recalls = []
            test_accuracies = []
            test_f1_scores = []
            test_precisions = []
            test_recalls = []

            for experiment in range(1, 11):  # 进行10次实验
                # Split the target batch into validation and test sets
                X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

                drca = DRCA(n_components=n_components, alpha=alpha)
                Xs_transformed, X_val_transformed = drca.fit_transform(Xs, X_val)
                Xt_transformed = drca.transform(X_test)

                model = create_model(Xs_transformed.shape[1])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(Xs_transformed, Ys, epochs=10, batch_size=32, verbose=0)

                # 验证集评估 (CN comment)
                val_loss, val_accuracy = model.evaluate(X_val_transformed, Y_val, verbose=0)
                val_accuracies.append(val_accuracy)

                Y_val_pred = model.predict(X_val_transformed)
                Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
                Y_val_classes = np.argmax(Y_val, axis=1)
                val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

                val_f1_scores.append(val_f1)
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)

                # 测试集评估 (CN comment)
                test_loss, test_accuracy = model.evaluate(Xt_transformed, Y_test, verbose=0)
                test_accuracies.append(test_accuracy)

                Y_test_pred = model.predict(Xt_transformed)
                Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
                Y_test_classes = np.argmax(Y_test, axis=1)
                test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

                test_f1_scores.append(test_f1)
                test_precisions.append(test_precision)
                test_recalls.append(test_recall)

                detailed_accuracies.append({'Model': f'drca_model_{j}', 'Experiment': f'n_components={n_components}, alpha={alpha}', 'Val Accuracy': val_accuracy, 'Val F1 Score': val_f1, 'Val Precision': val_precision, 'Val Recall': val_recall, 'Test Accuracy': test_accuracy, 'Test F1 Score': test_f1, 'Test Precision': test_precision, 'Test Recall': test_recall, 'Repeat': experiment})

            avg_val_accuracy = np.mean(val_accuracies)
            avg_val_f1 = np.mean(val_f1_scores)
            avg_val_precision = np.mean(val_precisions)
            avg_val_recall = np.mean(val_recalls)
            avg_test_accuracy = np.mean(test_accuracies)
            avg_test_f1 = np.mean(test_f1_scores)
            avg_test_precision = np.mean(test_precisions)
            avg_test_recall = np.mean(test_recalls)

            if avg_val_accuracy > best_accuracy:
                best_accuracy = avg_val_accuracy
                best_n_components = n_components
                best_alpha = alpha
                best_val_metrics = [{'Val Accuracy': a, 'Val F1 Score': f, 'Val Precision': p, 'Val Recall': r} for a, f, p, r in zip(val_accuracies, val_f1_scores, val_precisions, val_recalls)]
                best_test_metrics = [{'Test Accuracy': a, 'Test F1 Score': f, 'Test Precision': p, 'Test Recall': r} for a, f, p, r in zip(test_accuracies, test_f1_scores, test_precisions, test_recalls)]

    # Record the best result
    summary_results = {
        'Model': f'drca_model_{j}',
        'Best n_components': best_n_components,
        'Best Alpha': best_alpha,
        'Val Accuracy': [m['Val Accuracy'] for m in best_val_metrics],
        'Val F1 Score': [m['Val F1 Score'] for m in best_val_metrics],
        'Val Precision': [m['Val Precision'] for m in best_val_metrics],
        'Val Recall': [m['Val Recall'] for m in best_val_metrics],
        'Test Accuracy': [m['Test Accuracy'] for m in best_test_metrics],
        'Test F1 Score': [m['Test F1 Score'] for m in best_test_metrics],
        'Test Precision': [m['Test Precision'] for m in best_test_metrics],
        'Test Recall': [m['Test Recall'] for m in best_test_metrics],
        'Avg Val Accuracy': best_accuracy,
        'Avg Val F1 Score': avg_val_f1,
        'Avg Val Precision': avg_val_precision,
        'Avg Val Recall': avg_val_recall,
        'Avg Test Accuracy': avg_test_accuracy,
        'Avg Test F1 Score': avg_test_f1,
        'Avg Test Precision': avg_test_precision,
        'Avg Test Recall': avg_test_recall
    }

    results.append(summary_results)

# Convert results to DataFrame
detailed_results_df = pd.DataFrame(detailed_accuracies)
summary_results_df = pd.DataFrame(results)

# Save detailed results to an Excel file
detailed_save_path = "outputs/DRCA一对一详细.xlsx"
detailed_results_df.to_excel(detailed_save_path, index=False)

# 保存汇总结果到Excel文件 (CN comment)
summary_save_path = "outputs/DRCA一对一汇总.xlsx"
summary_results_df.to_excel(summary_save_path, index=False)

# Print summary results进行检查 (CN comment)
print(summary_results_df)

#30遍 (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

# 从汇总文件读取最佳条件并进行30次重复实验 (CN comment)
for j in range(2, 11):
    summary_path = f"outputs/DRCA一对一汇总.xlsx"
    summary_df = pd.read_excel(summary_path)

    best_n_components = summary_df.loc[summary_df['Model'] == f'drca_model_{j}', 'Best n_components'].values[0]
    best_alpha = summary_df.loc[summary_df['Model'] == f'drca_model_{j}', 'Best Alpha'].values[0]

    results = []
    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs, Ys = all_data[0]  # batch 1的数据作为源域数据

    for experiment in range(1, 31):  # 进行30次实验
        X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

        drca = DRCA(n_components=best_n_components, alpha=best_alpha)
        Xs_transformed, X_val_transformed = drca.fit_transform(Xs, X_val)
        Xt_transformed = drca.transform(X_test)

        model = create_model(Xs_transformed.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(Xs_transformed, Ys, epochs=10, batch_size=32, verbose=0)

        # 验证集评估 (CN comment)
        val_loss, val_accuracy = model.evaluate(X_val_transformed, Y_val, verbose=0)
        Y_val_pred = model.predict(X_val_transformed)
        Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
        Y_val_classes = np.argmax(Y_val, axis=1)
        val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

        # 测试集评估 (CN comment)
        test_loss, test_accuracy = model.evaluate(Xt_transformed, Y_test, verbose=0)
        Y_test_pred = model.predict(Xt_transformed)
        Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
        Y_test_classes = np.argmax(Y_test, axis=1)
        test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

        detailed_accuracies.append({
            'Model': f'drca_model_{j}', 
            'Experiment': experiment, 
            'n_components': best_n_components, 
            'Alpha': best_alpha, 
            'Val Accuracy': val_accuracy, 
            'Val F1 Score': val_f1, 
            'Val Precision': val_precision, 
            'Val Recall': val_recall, 
            'Test Accuracy': test_accuracy, 
            'Test F1 Score': test_f1, 
            'Test Precision': test_precision, 
            'Test Recall': test_recall
        })

    # Convert results to DataFrame
    detailed_results_df = pd.DataFrame(detailed_accuracies)

    # Save detailed results to an Excel file
    detailed_save_path = f"outputs/DRCA一对一详细_batch_{j}_30遍.xlsx"
    detailed_results_df.to_excel(detailed_save_path, index=False)

    # 打印完成信息 (CN comment)
    print(f"Batch {j} - 30 repetitions completed and saved to {detailed_save_path}")

# KD和DRCA结合 (CN comment)

#KD_DRCA结合 8.1 3参定制版 (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# Preprocess and cache all batches
all_data = []
for i in range(1, 11):
    data = process_data(f'batch{i}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for i in range(10):
    end = start + all_data[i][0].shape[0]
    all_data[i] = (X_combined_scaled[start:end], all_data[i][1])
    start = end

n_components_options = [50, 100, 150, 200, 300, 500]
alpha_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
temperature_options = [0.3, 1, 2, 3, 5, 25, 50, 100, 200]

for j in range(9, 11):
    results = []
    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs = all_data[0][0]
    Ys = all_data[0][1]

    best_accuracy = 0
    best_n_components = None
    best_alpha = None
    best_temperature = None
    best_val_metrics = []
    best_test_metrics = []

    for n_components in n_components_options:
        for alpha in alpha_options:
            for temperature in temperature_options:
                val_accuracies = []
                val_f1_scores = []
                val_precisions = []
                val_recalls = []
                test_accuracies = []
                test_f1_scores = []
                test_precisions = []
                test_recalls = []

                for experiment in range(1, 4):  # 进行3次实验
                    # Split the target batch into validation and test sets
                    X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

                    # DRCA拟合 (CN comment)
                    drca = DRCA(n_components=n_components, alpha=alpha)
                    Xs_transformed, X_val_transformed = drca.fit_transform(Xs, X_val)
                    Xt_transformed = drca.transform(X_test)

                    input_dim = Xs_transformed.shape[1]
                    teacher_model = create_model(input_dim)
                    student_model = create_model(input_dim)
                    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, temperature), metrics=['accuracy'])
                    teacher_model.fit(Xs_transformed, Ys, epochs=15, batch_size=32, verbose=0, validation_data=(X_val_transformed, Y_val))
                    soft_labels_train = teacher_model.predict(Xs_transformed)
                    soft_labels_val = teacher_model.predict(X_val_transformed)

                    # Merge soft labels
                    soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
                    X_combined_with_val = np.vstack((Xs_transformed, X_val_transformed))

                    student_model.fit(X_combined_with_val, soft_labels_combined, epochs=15, batch_size=32, verbose=0)

                    # Evaluate on the validation set
                    val_loss, val_accuracy = student_model.evaluate(X_val_transformed, Y_val, verbose=0)
                    val_accuracies.append(val_accuracy)

                    Y_val_pred = student_model.predict(X_val_transformed)
                    Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
                    Y_val_classes = np.argmax(Y_val, axis=1)
                    val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                    val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                    val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

                    val_f1_scores.append(val_f1)
                    val_precisions.append(val_precision)
                    val_recalls.append(val_recall)

                    # Evaluate on the test set
                    test_loss, test_accuracy = student_model.evaluate(Xt_transformed, Y_test, verbose=0)
                    test_accuracies.append(test_accuracy)

                    Y_test_pred = student_model.predict(Xt_transformed)
                    Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
                    Y_test_classes = np.argmax(Y_test, axis=1)
                    test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                    test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                    test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

                    test_f1_scores.append(test_f1)
                    test_precisions.append(test_precision)
                    test_recalls.append(test_recall)

                    detailed_accuracies.append({'Model': f'student_model_{j}', 'Experiment': experiment, 'n_components': n_components, 'Alpha': alpha, 'Temperature': temperature, 'Val Accuracy': val_accuracy, 'Val F1 Score': val_f1, 'Val Precision': val_precision, 'Val Recall': val_recall, 'Test Accuracy': test_accuracy, 'Test F1 Score': test_f1, 'Test Precision': test_precision, 'Test Recall': test_recall})

                avg_val_accuracy = np.mean(val_accuracies)
                avg_val_f1 = np.mean(val_f1_scores)
                avg_val_precision = np.mean(val_precisions)
                avg_val_recall = np.mean(val_recalls)

                avg_test_accuracy = np.mean(test_accuracies)
                avg_test_f1 = np.mean(test_f1_scores)
                avg_test_precision = np.mean(test_precisions)
                avg_test_recall = np.mean(test_recalls)

                if avg_val_accuracy > best_accuracy:
                    best_accuracy = avg_val_accuracy
                    best_n_components = n_components
                    best_alpha = alpha
                    best_temperature = temperature
                    best_val_metrics = [{'Val Accuracy': a, 'Val F1 Score': f, 'Val Precision': p, 'Val Recall': r} for a, f, p, r in zip(val_accuracies, val_f1_scores, val_precisions, val_recalls)]
                    best_test_metrics = [{'Test Accuracy': a, 'Test F1 Score': f, 'Test Precision': p, 'Test Recall': r} for a, f, p, r in zip(test_accuracies, test_f1_scores, test_precisions, test_recalls)]

    # Record the best result
    summary_results = {
        'Model': f'student_model_{j}',
        'Best n_components': best_n_components,
        'Best Alpha': best_alpha,
        'Best Temperature': best_temperature,
        'Val Accuracy': [m['Val Accuracy'] for m in best_val_metrics],
        'Val F1 Score': [m['Val F1 Score'] for m in best_val_metrics],
        'Val Precision': [m['Val Precision'] for m in best_val_metrics],
        'Val Recall': [m['Val Recall'] for m in best_val_metrics],
        'Test Accuracy': [m['Test Accuracy'] for m in best_test_metrics],
        'Test F1 Score': [m['Test F1 Score'] for m in best_test_metrics],
        'Test Precision': [m['Test Precision'] for m in best_test_metrics],
        'Test Recall': [m['Test Recall'] for m in best_test_metrics],
        'Avg Val Accuracy': best_accuracy,
        'Avg Val F1 Score': avg_val_f1,
        'Avg Val Precision': avg_val_precision,
        'Avg Val Recall': avg_val_recall,
        'Avg Test Accuracy': avg_test_accuracy,
        'Avg Test F1 Score': avg_test_f1,
        'Avg Test Precision': avg_test_precision,
        'Avg Test Recall': avg_test_recall
    }

    results.append(summary_results)

    # Convert results to DataFrame
    detailed_results_df = pd.DataFrame(detailed_accuracies)
    summary_results_df = pd.DataFrame(results)

    # Save detailed results to an Excel file
    detailed_save_path = f"outputs/DRCA_KD一对一详细_batch_{j}.xlsx"
    detailed_results_df.to_excel(detailed_save_path, index=False)

    # 保存汇总结果到Excel文件 (CN comment)
    summary_save_path = f"outputs/DRCA_KD一对一汇总_batch_{j}.xlsx"
    summary_results_df.to_excel(summary_save_path, index=False)

    # Print summary results进行检查 (CN comment)
    print(f"Batch {j} summary results:")
    print(summary_results_df)

#重复30最佳 (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# Preprocess and cache all batches
all_data = []
for i in range(1, 11):
    data = process_data(f'batch{i}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Split standardized data back into batches
start = 0
for i in range(10):
    end = start + all_data[i][0].shape[0]
    all_data[i] = (X_combined_scaled[start:end], all_data[i][1])
    start = end

# 读取最佳条件并进行30次重复实验 (CN comment)
for j in range(2, 11):
    # 从汇总文件读取最佳条件 (CN comment)
    summary_path = f"outputs/DRCA_KD一对一汇总_batch_{j}.xlsx"
    summary_df = pd.read_excel(summary_path)
    
    best_n_components = summary_df['Best n_components'].values[0]
    best_alpha = summary_df['Best Alpha'].values[0]
    best_temperature = summary_df['Best Temperature'].values[0]
    
    results = []
    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs = all_data[0][0]
    Ys = all_data[0][1]

    for experiment in range(1, 31):  # 进行30次实验
        # Split the target batch into validation and test sets
        X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

        # DRCA拟合 (CN comment)
        drca = DRCA(n_components=best_n_components, alpha=best_alpha)
        Xs_transformed, X_val_transformed = drca.fit_transform(Xs, X_val)
        Xt_transformed = drca.transform(X_test)

        input_dim = Xs_transformed.shape[1]
        teacher_model = create_model(input_dim)
        student_model = create_model(input_dim)
        teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, best_temperature), metrics=['accuracy'])
        teacher_model.fit(Xs_transformed, Ys, epochs=15, batch_size=32, verbose=0, validation_data=(X_val_transformed, Y_val))
        soft_labels_train = teacher_model.predict(Xs_transformed)
        soft_labels_val = teacher_model.predict(X_val_transformed)

        # Merge soft labels
        soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
        X_combined_with_val = np.vstack((Xs_transformed, X_val_transformed))

        student_model.fit(X_combined_with_val, soft_labels_combined, epochs=15, batch_size=32, verbose=0)

        # Evaluate on the validation set
        val_loss, val_accuracy = student_model.evaluate(X_val_transformed, Y_val, verbose=0)
        Y_val_pred = student_model.predict(X_val_transformed)
        Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
        Y_val_classes = np.argmax(Y_val, axis=1)
        val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
        val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

        # Evaluate on the test set
        test_loss, test_accuracy = student_model.evaluate(Xt_transformed, Y_test, verbose=0)
        Y_test_pred = student_model.predict(Xt_transformed)
        Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
        Y_test_classes = np.argmax(Y_test, axis=1)
        test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
        test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)

        detailed_accuracies.append({
            'Model': f'student_model_{j}', 
            'Experiment': experiment, 
            'n_components': best_n_components, 
            'Alpha': best_alpha, 
            'Temperature': best_temperature, 
            'Val Accuracy': val_accuracy, 
            'Val F1 Score': val_f1, 
            'Val Precision': val_precision, 
            'Val Recall': val_recall, 
            'Test Accuracy': test_accuracy, 
            'Test F1 Score': test_f1, 
            'Test Precision': test_precision, 
            'Test Recall': test_recall
        })

    # Convert results to DataFrame
    detailed_results_df = pd.DataFrame(detailed_accuracies)

    # Save detailed results to an Excel file
    detailed_save_path = f"outputs/DRCA_KD一对一详细_batch_{j}_30遍.xlsx"
    detailed_results_df.to_excel(detailed_save_path, index=False)

    # 打印完成信息 (CN comment)
    print(f"Batch {j} - 30 repetitions completed and saved to {detailed_save_path}")

#accuracy做图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_summary_path = "outputs/知识蒸馏一对一汇总test.xlsx"
drca_summary_path = "outputs/DRCA一对一汇总.xlsx"
kd_drca_base_path = "outputs/"

def process_data(file_path, model_prefix, test_metric, val_metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    if test_metric in df.columns:
        df[test_metric] = df[test_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(test_metric)
        df_test = df_test.rename(columns={test_metric: 'Accuracy'})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    if val_metric in df.columns:
        df[val_metric] = df[val_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(val_metric)
        df_val = df_val.rename(columns={val_metric: 'Accuracy'})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

# Load and preprocess data
baseline_test, baseline_val = process_data(baseline_path, 'model_', 'Test Accuracy', 'Val Accuracy')
baseline_test['Method'] = 'Baseline'
baseline_val['Method'] = 'Baseline'

kd_test, kd_val = process_data(kd_summary_path, 'student_model_', 'Test Accuracy', 'Val Accuracy')
kd_test['Method'] = 'Knowledge Distillation'
kd_val['Method'] = 'Knowledge Distillation'

drca_test, drca_val = process_data(drca_summary_path, 'drca_model_', 'Test Accuracy', 'Val Accuracy')
drca_test['Method'] = 'DRCA'
drca_val['Method'] = 'DRCA'

kd_drca_test_list = []
kd_drca_val_list = []
for batch_num in range(2, 11):  # 假设有10个batch
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一汇总_batch_{batch_num}.xlsx')
    kd_drca_batch_test, kd_drca_batch_val = process_data(batch_file_path, 'student_model_', 'Test Accuracy', 'Val Accuracy')
    kd_drca_batch_test['Method'] = 'KD-DRCA'
    kd_drca_batch_val['Method'] = 'KD-DRCA'
    kd_drca_batch_test['Batch'] = batch_num  # 添加Batch列
    kd_drca_batch_val['Batch'] = batch_num  # 添加Batch列
    kd_drca_test_list.append(kd_drca_batch_test)
    kd_drca_val_list.append(kd_drca_batch_val)

kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

# 合并测试集数据 (CN comment)
df_combined_accuracy_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

# 合并验证集数据 (CN comment)
df_combined_accuracy_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

# Plot boxplots on the test set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Accuracy', data=df_combined_accuracy_test, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Test batch', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save test plots and data to files
output_path_test = "outputs/"
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)
plt.savefig(output_path_test + "accuracy_test.png")

# Display test plots
plt.tight_layout()
plt.show()

# Save test data
df_combined_accuracy_test.to_excel(output_path_test + "accuracy_test_data.xlsx", index=False)

# Print test data output path
print("测试集数据已保存到:", output_path_test + "accuracy_test_data.xlsx")
print("测试集图表已保存到:", output_path_test + "accuracy_test.png")

# Plot boxplots on the validation set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Accuracy', data=df_combined_accuracy_val, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Validation batch', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save validation plots and data to files
output_path_val = "outputs/"
if not os.path.exists(output_path_val):
    os.makedirs(output_path_val)
plt.savefig(output_path_val + "accuracy_val.png")

# Display validation plots
plt.tight_layout()
plt.show()

# Save validation data
df_combined_accuracy_val.to_excel(output_path_val + "accuracy_val_data.xlsx", index=False)

# Print validation data output path
print("验证集数据已保存到:", output_path_val + "accuracy_val_data.xlsx")
print("验证集图表已保存到:", output_path_val + "accuracy_val.png")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_base_path = "outputs/"
drca_base_path = "outputs/"
kd_drca_base_path = "outputs/"

def process_data(file_path, model_prefix, test_metric, val_metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    if test_metric in df.columns:
        df[test_metric] = df[test_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(test_metric)
        df_test = df_test.rename(columns={test_metric: 'Accuracy'})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    if val_metric in df.columns:
        df[val_metric] = df[val_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(val_metric)
        df_val = df_val.rename(columns={val_metric: 'Accuracy'})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

# Load and preprocess data
baseline_test, baseline_val = process_data(baseline_path, 'model_', 'Test Accuracy', 'Val Accuracy')
baseline_test['Method'] = 'Baseline'
baseline_val['Method'] = 'Baseline'

kd_test_list = []
kd_val_list = []
drca_test_list = []
drca_val_list = []
kd_drca_test_list = []
kd_drca_val_list = []

for n in range(2, 11):  # n的范围是2～10
    kd_file_path = os.path.join(kd_base_path, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
    drca_file_path = os.path.join(drca_base_path, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
    kd_drca_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')
    
    kd_batch_test, kd_batch_val = process_data(kd_file_path, 'student_model_', 'Test Accuracy', 'Val Accuracy')
    kd_batch_test['Method'] = 'Knowledge Distillation'
    kd_batch_val['Method'] = 'Knowledge Distillation'
    kd_batch_test['Batch'] = n
    kd_batch_val['Batch'] = n
    kd_test_list.append(kd_batch_test)
    kd_val_list.append(kd_batch_val)
    
    drca_batch_test, drca_batch_val = process_data(drca_file_path, 'drca_model_', 'Test Accuracy', 'Val Accuracy')
    drca_batch_test['Method'] = 'DRCA'
    drca_batch_val['Method'] = 'DRCA'
    drca_batch_test['Batch'] = n
    drca_batch_val['Batch'] = n
    drca_test_list.append(drca_batch_test)
    drca_val_list.append(drca_batch_val)
    
    kd_drca_batch_test, kd_drca_batch_val = process_data(kd_drca_file_path, 'student_model_', 'Test Accuracy', 'Val Accuracy')
    kd_drca_batch_test['Method'] = 'KD-DRCA'
    kd_drca_batch_val['Method'] = 'KD-DRCA'
    kd_drca_batch_test['Batch'] = n
    kd_drca_batch_val['Batch'] = n
    kd_drca_test_list.append(kd_drca_batch_test)
    kd_drca_val_list.append(kd_drca_batch_val)

kd_test = pd.concat(kd_test_list, ignore_index=True)
kd_val = pd.concat(kd_val_list, ignore_index=True)
drca_test = pd.concat(drca_test_list, ignore_index=True)
drca_val = pd.concat(drca_val_list, ignore_index=True)
kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

# 合并测试集数据 (CN comment)
df_combined_accuracy_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

# 合并验证集数据 (CN comment)
df_combined_accuracy_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

# Plot boxplots on the test set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Accuracy', data=df_combined_accuracy_test, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Test batch', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)

# Save test plots and data to files
output_path_test = "outputs/"
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)
plt.savefig(output_path_test + "accuracy_test_30遍.png")

# Display test plots
plt.tight_layout()
plt.show()

# Save test data
df_combined_accuracy_test.to_excel(output_path_test + "accuracy_test_data_30遍.xlsx", index=False)

# Print test data output path
print("测试集数据已保存到:", output_path_test + "accuracy_test_data_30遍.xlsx")
print("测试集图表已保存到:", output_path_test + "accuracy_test_30遍.png")

# Plot boxplots on the validation set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Accuracy', data=df_combined_accuracy_val, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Validation batch', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)

# Save validation plots and data to files
output_path_val = "outputs/"
if not os.path.exists(output_path_val):
    os.makedirs(output_path_val)
plt.savefig(output_path_val + "accuracy_val_30遍.png")

# Display validation plots
plt.tight_layout()
plt.show()

# Save validation data
df_combined_accuracy_val.to_excel(output_path_val + "accuracy_val_data_30遍.xlsx", index=False)

# Print validation data output path
print("验证集数据已保存到:", output_path_val + "accuracy_val_data_30遍.xlsx")
print("验证集图表已保存到:", output_path_val + "accuracy_val_30遍.png")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_base_path = "outputs/"
drca_base_path = "outputs/"
kd_drca_base_path = "outputs/"

metrics = ['F1 Score', 'Recall', 'Precision']

def process_data(file_path, model_prefix, metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    metric_key = f'Test {metric}'
    if metric_key in df.columns:
        df[metric_key] = df[metric_key].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(metric_key)
        df_test = df_test.rename(columns={metric_key: metric})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    metric_key_val = f'Val {metric}'
    if metric_key_val in df.columns:
        df[metric_key_val] = df[metric_key_val].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(metric_key_val)
        df_val = df_val.rename(columns={metric_key_val: metric})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

for metric in metrics:
    # Load and preprocess data
    baseline_test, baseline_val = process_data(baseline_path, 'model_', metric)
    baseline_test['Method'] = 'Baseline'
    baseline_val['Method'] = 'Baseline'

    kd_test_list = []
    kd_val_list = []
    drca_test_list = []
    drca_val_list = []
    kd_drca_test_list = []
    kd_drca_val_list = []

    for n in range(2, 11):  # n的范围是2～10
        kd_file_path = os.path.join(kd_base_path, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
        drca_file_path = os.path.join(drca_base_path, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
        kd_drca_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')
        
        kd_batch_test, kd_batch_val = process_data(kd_file_path, 'student_model_', metric)
        kd_batch_test['Method'] = 'Knowledge Distillation'
        kd_batch_val['Method'] = 'Knowledge Distillation'
        kd_batch_test['Batch'] = n
        kd_batch_val['Batch'] = n
        kd_test_list.append(kd_batch_test)
        kd_val_list.append(kd_batch_val)
        
        drca_batch_test, drca_batch_val = process_data(drca_file_path, 'drca_model_', metric)
        drca_batch_test['Method'] = 'DRCA'
        drca_batch_val['Method'] = 'DRCA'
        drca_batch_test['Batch'] = n
        drca_batch_val['Batch'] = n
        drca_test_list.append(drca_batch_test)
        drca_val_list.append(drca_batch_val)
        
        kd_drca_batch_test, kd_drca_batch_val = process_data(kd_drca_file_path, 'student_model_', metric)
        kd_drca_batch_test['Method'] = 'KD-DRCA'
        kd_drca_batch_val['Method'] = 'KD-DRCA'
        kd_drca_batch_test['Batch'] = n
        kd_drca_batch_val['Batch'] = n
        kd_drca_test_list.append(kd_drca_batch_test)
        kd_drca_val_list.append(kd_drca_batch_val)

    kd_test = pd.concat(kd_test_list, ignore_index=True)
    kd_val = pd.concat(kd_val_list, ignore_index=True)
    drca_test = pd.concat(drca_test_list, ignore_index=True)
    drca_val = pd.concat(drca_val_list, ignore_index=True)
    kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
    kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

    # 合并测试集数据 (CN comment)
    df_combined_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

    # 合并验证集数据 (CN comment)
    df_combined_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

    # Plot boxplots on the test set
    plt.figure(figsize=(18, 10), dpi=80)
    sns.boxplot(x='Model', y=metric, data=df_combined_test, hue='Method')

    # Adjust X/Y-axis label font size
    plt.xlabel('Test batch', fontsize=32)
    plt.ylabel(metric, fontsize=32)

    # Adjust X/Y-axis tick font size
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Adjust legend position and font size
    plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

    # Add grid
    plt.grid(True)

    # Save test plots and data to files
    output_path_test = f"outputs/"
    if not os.path.exists(output_path_test):
        os.makedirs(output_path_test)
    plt.savefig(output_path_test + f"{metric.lower()}_test_30遍.png")

    # Display test plots
    plt.tight_layout()
    plt.show()

    # Save test data
    df_combined_test.to_excel(output_path_test + f"{metric.lower()}_test_data_30遍.xlsx", index=False)

    # Print test data output path
    print(f"测试集数据已保存到: {output_path_test}{metric.lower()}_test_data_30遍.xlsx")
    print(f"测试集图表已保存到: {output_path_test}{metric.lower()}_test_30遍.png")

    # Plot boxplots on the validation set
    plt.figure(figsize=(18, 10), dpi=80)
    sns.boxplot(x='Model', y=metric, data=df_combined_val, hue='Method')

    # Adjust X/Y-axis label font size
    plt.xlabel('Validation batch', fontsize=32)
    plt.ylabel(metric, fontsize=32)

    # Adjust X/Y-axis tick font size
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Adjust legend position and font size
    plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

    # Add grid
    plt.grid(True)

    # Save validation plots and data to files
    output_path_val = f"outputs/"
    if not os.path.exists(output_path_val):
        os.makedirs(output_path_val)
    plt.savefig(output_path_val + f"{metric.lower()}_val_30遍.png")

    # Display validation plots
    plt.tight_layout()
    plt.show()

    # Save validation data
    df_combined_val.to_excel(output_path_val + f"{metric.lower()}_val_data_30遍.xlsx", index=False)

    # Print validation data output path
    print(f"验证集数据已保存到: {output_path_val}{metric.lower()}_val_data_30遍.xlsx")
    print(f"验证集图表已保存到: {output_path_val}{metric.lower()}_val_30遍.png")

#macro f1 score做图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_summary_path = "outputs/知识蒸馏一对一汇总test.xlsx"
drca_summary_path = "outputs/DRCA一对一汇总.xlsx"
kd_drca_base_path = "outputs/"

def process_data(file_path, model_prefix, test_metric, val_metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    if test_metric in df.columns:
        df[test_metric] = df[test_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(test_metric)
        df_test = df_test.rename(columns={test_metric: 'F1 Score'})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    if val_metric in df.columns:
        df[val_metric] = df[val_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(val_metric)
        df_val = df_val.rename(columns={val_metric: 'F1 Score'})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

# Load and preprocess data
baseline_test, baseline_val = process_data(baseline_path, 'model_', 'Test F1 Score', 'Val F1 Score')
baseline_test['Method'] = 'Baseline'
baseline_val['Method'] = 'Baseline'

kd_test, kd_val = process_data(kd_summary_path, 'student_model_', 'Test F1 Score', 'Val F1 Score')
kd_test['Method'] = 'Knowledge Distillation'
kd_val['Method'] = 'Knowledge Distillation'

drca_test, drca_val = process_data(drca_summary_path, 'drca_model_', 'Test F1 Score', 'Val F1 Score')
drca_test['Method'] = 'DRCA'
drca_val['Method'] = 'DRCA'

kd_drca_test_list = []
kd_drca_val_list = []
for batch_num in range(2, 11):  # 假设有10个batch
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一汇总_batch_{batch_num}.xlsx')
    kd_drca_batch_test, kd_drca_batch_val = process_data(batch_file_path, 'student_model_', 'Test F1 Score', 'Val F1 Score')
    kd_drca_batch_test['Method'] = 'KD-DRCA'
    kd_drca_batch_val['Method'] = 'KD-DRCA'
    kd_drca_batch_test['Batch'] = batch_num  # 添加Batch列
    kd_drca_batch_val['Batch'] = batch_num  # 添加Batch列
    kd_drca_test_list.append(kd_drca_batch_test)
    kd_drca_val_list.append(kd_drca_batch_val)

kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

# 合并测试集数据 (CN comment)
df_combined_f1_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

# 合并验证集数据 (CN comment)
df_combined_f1_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

# Plot boxplots on the test set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='F1 Score', data=df_combined_f1_test, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Test batch', fontsize=32)
plt.ylabel('F1 Score', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save test plots and data to files
output_path_test = "outputs/"
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)
plt.savefig(output_path_test + "f1_score_test.png")

# Display test plots
plt.tight_layout()
plt.show()

# Save test data
df_combined_f1_test.to_excel(output_path_test + "f1_score_test_data.xlsx", index=False)

# Print test data output path
print("测试集数据已保存到:", output_path_test + "f1_score_test_data.xlsx")
print("测试集图表已保存到:", output_path_test + "f1_score_test.png")

# Plot boxplots on the validation set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='F1 Score', data=df_combined_f1_val, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Validation batch', fontsize=32)
plt.ylabel('F1 Score', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save validation plots and data to files
output_path_val = "outputs/"
if not os.path.exists(output_path_val):
    os.makedirs(output_path_val)
plt.savefig(output_path_val + "f1_score_val.png")

# Display validation plots
plt.tight_layout()
plt.show()

# Save validation data
df_combined_f1_val.to_excel(output_path_val + "f1_score_val_data.xlsx", index=False)

# Print validation data output path
print("验证集数据已保存到:", output_path_val + "f1_score_val_data.xlsx")
print("验证集图表已保存到:", output_path_val + "f1_score_val.png")

#recall绘图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_summary_path = "outputs/知识蒸馏一对一汇总test.xlsx"
drca_summary_path = "outputs/DRCA一对一汇总.xlsx"
kd_drca_base_path = "outputs/"

def process_data(file_path, model_prefix, test_metric, val_metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    if test_metric in df.columns:
        df[test_metric] = df[test_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(test_metric)
        df_test = df_test.rename(columns={test_metric: 'Recall'})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    if val_metric in df.columns:
        df[val_metric] = df[val_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(val_metric)
        df_val = df_val.rename(columns={val_metric: 'Recall'})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

# Load and preprocess data
baseline_test, baseline_val = process_data(baseline_path, 'model_', 'Test Recall', 'Val Recall')
baseline_test['Method'] = 'Baseline'
baseline_val['Method'] = 'Baseline'

kd_test, kd_val = process_data(kd_summary_path, 'student_model_', 'Test Recall', 'Val Recall')
kd_test['Method'] = 'Knowledge Distillation'
kd_val['Method'] = 'Knowledge Distillation'

drca_test, drca_val = process_data(drca_summary_path, 'drca_model_', 'Test Recall', 'Val Recall')
drca_test['Method'] = 'DRCA'
drca_val['Method'] = 'DRCA'

kd_drca_test_list = []
kd_drca_val_list = []
for batch_num in range(2, 11):  # 假设有10个batch
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一汇总_batch_{batch_num}.xlsx')
    kd_drca_batch_test, kd_drca_batch_val = process_data(batch_file_path, 'student_model_', 'Test Recall', 'Val Recall')
    kd_drca_batch_test['Method'] = 'KD-DRCA'
    kd_drca_batch_val['Method'] = 'KD-DRCA'
    kd_drca_batch_test['Batch'] = batch_num  # 添加Batch列
    kd_drca_batch_val['Batch'] = batch_num  # 添加Batch列
    kd_drca_test_list.append(kd_drca_batch_test)
    kd_drca_val_list.append(kd_drca_batch_val)

kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

# 合并测试集数据 (CN comment)
df_combined_recall_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

# 合并验证集数据 (CN comment)
df_combined_recall_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

# Plot boxplots on the test set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Recall', data=df_combined_recall_test, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Test batch', fontsize=32)
plt.ylabel('Recall', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save test plots and data to files
output_path_test = "outputs/"
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)
plt.savefig(output_path_test + "recall_test.png")

# Display test plots
plt.tight_layout()
plt.show()

# Save test data
df_combined_recall_test.to_excel(output_path_test + "recall_test_data.xlsx", index=False)

# Print test data output path
print("测试集数据已保存到:", output_path_test + "recall_test_data.xlsx")
print("测试集图表已保存到:", output_path_test + "recall_test.png")

# Plot boxplots on the validation set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Recall', data=df_combined_recall_val, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Validation batch', fontsize=32)
plt.ylabel('Recall', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save validation plots and data to files
output_path_val = "outputs/"
if not os.path.exists(output_path_val):
    os.makedirs(output_path_val)
plt.savefig(output_path_val + "recall_val.png")

# Display validation plots
plt.tight_layout()
plt.show()

# Save validation data
df_combined_recall_val.to_excel(output_path_val + "recall_val_data.xlsx", index=False)

# Print validation data output path
print("验证集数据已保存到:", output_path_val + "recall_val_data.xlsx")
print("验证集图表已保存到:", output_path_val + "recall_val.png")

#precision做图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/一对一详细.xlsx"
kd_summary_path = "outputs/知识蒸馏一对一汇总test.xlsx"
drca_summary_path = "outputs/DRCA一对一汇总.xlsx"
kd_drca_base_path = "outputs/"

def process_data(file_path, model_prefix, test_metric, val_metric):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)

    if test_metric in df.columns:
        df[test_metric] = df[test_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_test = df.explode(test_metric)
        df_test = df_test.rename(columns={test_metric: 'Precision'})
        df_test['Dataset'] = 'Test'
    else:
        df_test = pd.DataFrame()

    if val_metric in df.columns:
        df[val_metric] = df[val_metric].apply(lambda x: list(map(float, str(x).strip('[]').split(', '))) if isinstance(x, str) else [x])
        df_val = df.explode(val_metric)
        df_val = df_val.rename(columns={val_metric: 'Precision'})
        df_val['Dataset'] = 'Validation'
    else:
        df_val = pd.DataFrame()

    return df_test, df_val

# Load and preprocess data
baseline_test, baseline_val = process_data(baseline_path, 'model_', 'Test Precision', 'Val Precision')
baseline_test['Method'] = 'Baseline'
baseline_val['Method'] = 'Baseline'

kd_test, kd_val = process_data(kd_summary_path, 'student_model_', 'Test Precision', 'Val Precision')
kd_test['Method'] = 'Knowledge Distillation'
kd_val['Method'] = 'Knowledge Distillation'

drca_test, drca_val = process_data(drca_summary_path, 'drca_model_', 'Test Precision', 'Val Precision')
drca_test['Method'] = 'DRCA'
drca_val['Method'] = 'DRCA'

kd_drca_test_list = []
kd_drca_val_list = []
for batch_num in range(2, 11):  # 假设有10个batch
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD一对一汇总_batch_{batch_num}.xlsx')
    kd_drca_batch_test, kd_drca_batch_val = process_data(batch_file_path, 'student_model_', 'Test Precision', 'Val Precision')
    kd_drca_batch_test['Method'] = 'KD-DRCA'
    kd_drca_batch_val['Method'] = 'KD-DRCA'
    kd_drca_batch_test['Batch'] = batch_num  # 添加Batch列
    kd_drca_batch_val['Batch'] = batch_num  # 添加Batch列
    kd_drca_test_list.append(kd_drca_batch_test)
    kd_drca_val_list.append(kd_drca_batch_val)

kd_drca_test = pd.concat(kd_drca_test_list, ignore_index=True)
kd_drca_val = pd.concat(kd_drca_val_list, ignore_index=True)

# 合并测试集数据 (CN comment)
df_combined_precision_test = pd.concat([baseline_test, kd_test, drca_test, kd_drca_test], ignore_index=True)

# 合并验证集数据 (CN comment)
df_combined_precision_val = pd.concat([baseline_val, kd_val, drca_val, kd_drca_val], ignore_index=True)

# Plot boxplots on the test set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Precision', data=df_combined_precision_test, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Test batch', fontsize=32)
plt.ylabel('Precision', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save test plots and data to files
output_path_test = "outputs/"
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)
plt.savefig(output_path_test + "precision_test.png")

# Display test plots
plt.tight_layout()
plt.show()

# Save test data
df_combined_precision_test.to_excel(output_path_test + "precision_test_data.xlsx", index=False)

# Print test data output path
print("测试集数据已保存到:", output_path_test + "precision_test_data.xlsx")
print("测试集图表已保存到:", output_path_test + "precision_test.png")

# Plot boxplots on the validation set
plt.figure(figsize=(18, 10), dpi=80)
sns.boxplot(x='Model', y='Precision', data=df_combined_precision_val, hue='Method')

# Adjust X/Y-axis label font size
plt.xlabel('Validation batch', fontsize=32)
plt.ylabel('Precision', fontsize=32)

# Adjust X/Y-axis tick font size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Adjust legend position and font size
plt.legend(title='Method', title_fontsize='24', fontsize='22', loc='lower left', bbox_to_anchor=(0.1, 0))

# Add grid
plt.grid(True)


# Save validation plots and data to files
output_path_val = "outputs/"
if not os.path.exists(output_path_val):
    os.makedirs(output_path_val)
plt.savefig(output_path_val + "precision_val.png")

# Display validation plots
plt.tight_layout()
plt.show()

# Save validation data
df_combined_precision_val.to_excel(output_path_val + "precision_val_data.xlsx", index=False)

# Print validation data output path
print("验证集数据已保存到:", output_path_val + "precision_val_data.xlsx")
print("验证集图表已保存到:", output_path_val + "precision_val.png")

#绘制显著性差异table (CN comment)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_paths = [
    "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx",
    "outputs/知识蒸馏一对一汇总test.xlsx"
]

drca_paths = [
    "outputs/DRCA已有的预测新的（定制版）汇总.xlsx",
    "outputs/DRCA一对一汇总.xlsx"
]

kd_drca_base_paths = [
    "outputs/",
    "outputs/"
]

metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision', 'Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = []
    for model in range(2, 11):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            results.append('No Data')
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results.append('Better')
            else:
                results.append('Worse')
        else:
            results.append('No Difference')
    return results

# 处理所有数据 (CN comment)
def process_all_data():
    data = {}
    for metric in metrics:
        data[metric] = {
            "Baseline": pd.concat([process_data(path, 'model_', [metric]) for path in baseline_paths], ignore_index=True),
            "KD": pd.concat([process_data(path, 'student_model_', [metric]) for path in kd_paths], ignore_index=True),
            "DRCA": pd.concat([process_data(path, 'drca_model_', [metric]) for path in drca_paths], ignore_index=True),
            "KD_DRCA": pd.concat([process_data(os.path.join(base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch}.xlsx' if base_path == kd_drca_base_paths[0] else f'DRCA_KD一对一汇总_batch_{batch}.xlsx'), 'student_model_', [metric])
                                  for base_path in kd_drca_base_paths for batch in range(2, 11)], ignore_index=True)
        }
    return data

# 创建最终结果表格 (CN comment)
def create_results_table():
    data = process_all_data()
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 设置表格数据框架 (CN comment)
    table_data = []
    for method in ['KD', 'DRCA', 'KD_DRCA']:
        for metric in metrics:
            baseline_data = data[metric]["Baseline"]
            method_data = data[metric][method]
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, metric.split()[1]] + results)
    
    # 绘制表格 (CN comment)
    columns = ['Method', 'Metric'] + [f'Test Batch {i}' for i in range(2, 11)]
    colors = {'Better': 'red', 'No Difference': 'blue', 'Worse': 'green', 'No Data': 'gray'}
    cell_colors = [[colors.get(val, 'white') for val in row] for row in table_data]

    table = ax.table(cellText=table_data, cellColours=cell_colors, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(2, 4)

    ax.axis('off')

    # 图例 (CN comment)
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[key], label=key) for key in colors]
    plt.legend(handles=legend_patches, loc='lower right')

    plt.title('Significance Comparison of Different Methods for Various Metrics', fontsize=18)
    output_path = "outputs/"
    plt.savefig(output_path + "significance_comparison.png")
    plt.show()

    # 保存数据为Excel文件 (CN comment)
    result_df = pd.DataFrame(table_data, columns=columns)
    result_df.to_excel(output_path + "significance_comparison_data.xlsx", index=False)

# 调用函数 (CN comment)
create_results_table()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_paths = [
    "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx",
    "outputs/知识蒸馏一对一汇总test.xlsx"
]

drca_paths = [
    "outputs/DRCA已有的预测新的（定制版）汇总.xlsx",
    "outputs/DRCA一对一汇总.xlsx"
]

kd_drca_base_paths = [
    "outputs/",
    "outputs/"
]

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = []
    for model in range(2, 11):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            results.append('No Data')
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results.append('Better')
            else:
                results.append('Worse')
        else:
            results.append('No Difference')
    return results

# 处理所有数据 (CN comment)
def process_all_data(metrics):
    data = {}
    for metric in metrics:
        data[metric] = {
            "Baseline": pd.concat([process_data(path, 'model_', [metric]) for path in baseline_paths], ignore_index=True),
            "KD": pd.concat([process_data(path, 'student_model_', [metric]) for path in kd_paths], ignore_index=True),
            "DRCA": pd.concat([process_data(path, 'drca_model_', [metric]) for path in drca_paths], ignore_index=True),
            "KD_DRCA": pd.concat([process_data(os.path.join(base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch}.xlsx' if base_path == kd_drca_base_paths[0] else f'DRCA_KD一对一汇总_batch_{batch}.xlsx'), 'student_model_', [metric])
                                  for base_path in kd_drca_base_paths for batch in range(2, 11)], ignore_index=True)
        }
    return data

# 创建最终结果表格 (CN comment)
def create_results_table(metrics, metrics_name):
    data = process_all_data(metrics)
    output_path = "outputs/"
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # 设置表格数据框架 (CN comment)
        table_data = []
        for method in ['KD', 'DRCA', 'KD_DRCA']:
            baseline_data = data[metric]["Baseline"]
            method_data = data[metric][method]
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, metric.split()[1]] + results)
        
        # 绘制表格 (CN comment)
        columns = ['Method', 'Metric'] + [f'Test Batch {i}' for i in range(2, 11)]
        colors = {'Better': 'red', 'No Difference': 'blue', 'Worse': 'green', 'No Data': 'gray'}
        cell_colors = [[colors.get(val, 'white') for val in row] for row in table_data]

        table = ax.table(cellText=table_data, cellColours=cell_colors, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(24)
        table.scale(2, 4)

        ax.axis('off')

        # 图例 (CN comment)
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=colors[key], label=key) for key in colors]
        plt.legend(handles=legend_patches, loc='lower right')

        plt.title(f'Significance Comparison of Different Methods for {metrics_name}', fontsize=18)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison.png")
        plt.show()

        # 保存数据为Excel文件 (CN comment)
        result_df = pd.DataFrame(table_data, columns=columns)
        result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison_data.xlsx", index=False)

# 调用函数生成图表 (CN comment)
create_results_table(test_metrics, 'Test Metrics')
create_results_table(val_metrics, 'Validation Metrics')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_paths = [
    "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx",
    "outputs/知识蒸馏一对一汇总test.xlsx"
]

drca_paths = [
    "outputs/DRCA已有的预测新的（定制版）汇总.xlsx",
    "outputs/DRCA一对一汇总.xlsx"
]

kd_drca_base_paths = [
    "outputs/",
    "outputs/"
]

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    for model in range(2, 20):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results["positive"] += 1
            else:
                results["negative"] += 1
        else:
            results["neutral"] += 1
        
        results["total"] += 1
    return results

# 处理所有数据 (CN comment)
def process_all_data(metrics):
    data = {}
    for metric in metrics:
        data[metric] = {
            "Baseline": pd.concat([process_data(path, 'model_', [metric], offset=0 if i == 0 else 9) for i, path in enumerate(baseline_paths)], ignore_index=True),
            "KD": pd.concat([process_data(path, 'student_model_', [metric], offset=0 if i == 0 else 9) for i, path in enumerate(kd_paths)], ignore_index=True),
            "DRCA": pd.concat([process_data(path, 'drca_model_', [metric], offset=0 if i == 0 else 9) for i, path in enumerate(drca_paths)], ignore_index=True),
            "KD_DRCA": pd.concat([process_data(os.path.join(base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch}.xlsx' if base_path == kd_drca_base_paths[0] else f'DRCA_KD一对一汇总_batch_{batch}.xlsx'), 'student_model_', [metric], offset=0 if base_path == kd_drca_base_paths[0] else 9)
                                  for base_path in kd_drca_base_paths for batch in range(2, 11)], ignore_index=True)
        }
    return data

# 创建最终结果表格 (CN comment)
def create_results_table(metrics, metrics_name):
    data = process_all_data(metrics)
    output_path = "outputs/"
    
    for metric in metrics:
        table_data = []
        detailed_data = []

        for method in ['KD', 'DRCA', 'KD_DRCA']:
            baseline_data = data[metric]["Baseline"]
            method_data = data[metric][method]
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, results["positive"], results["neutral"], results["negative"], results["total"]])
            
            # 添加详细数据到detailed_data (CN comment)
            for model in range(2, 20):
                baseline_scores = baseline_data[baseline_data['Model'] == model][metric].values
                method_scores = method_data[method_data['Model'] == model][metric].values
                detailed_data.append([method, metric, model, list(baseline_scores), list(method_scores)])

        # 绘制表格 (CN comment)
        columns = ['Method', '+ (p<0.05)', '= (p>0.05)', '- (p<0.05)', 'Total']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.title(f'Significance Comparison for {metric} ({metrics_name})', fontsize=14)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison.png")
        plt.show()

        # 保存汇总数据为Excel文件 (CN comment)
        result_df = pd.DataFrame(table_data, columns=columns)
        result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison_data.xlsx", index=False)
        
        # 保存详细数据为Excel文件 (CN comment)
        detailed_columns = ['Method', 'Metric', 'Model', 'Baseline Scores', 'Method Scores']
        detailed_df = pd.DataFrame(detailed_data, columns=detailed_columns)
        detailed_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_detailed_data.xlsx", index=False)

# 调用函数生成图表 (CN comment)
create_results_table(test_metrics, 'Test Metrics')
create_results_table(val_metrics, 'Validation Metrics')

#8.25

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_base_path_existing = "outputs/"
drca_base_path_existing = "outputs/"
kd_drca_base_path_existing = "outputs/"

kd_base_path_one_to_one = "outputs/"
drca_base_path_one_to_one = "outputs/"
kd_drca_base_path_one_to_one = "outputs/"

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    for model in range(2, 20):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results["positive"] += 1
            else:
                results["negative"] += 1
        else:
            results["neutral"] += 1
        
        results["total"] += 1
    return results

# 处理所有数据 (CN comment)
def process_all_data(metrics):
    data = {}
    for metric in metrics:
        kd_test_list, drca_test_list, kd_drca_test_list = [], [], []

        # 处理一对一的每个批次，编号保持2～10不变 (CN comment)
        for n in range(2, 11):  # n的范围是2～10
            kd_file_path_one_to_one = os.path.join(kd_base_path_one_to_one, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
            drca_file_path_one_to_one = os.path.join(drca_base_path_one_to_one, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_one_to_one = os.path.join(kd_drca_base_path_one_to_one, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')

            kd_batch_test_one_to_one = process_data(kd_file_path_one_to_one, 'student_model_', [metric], offset=0)
            drca_batch_test_one_to_one = process_data(drca_file_path_one_to_one, 'drca_model_', [metric], offset=0)
            kd_drca_batch_test_one_to_one = process_data(kd_drca_file_path_one_to_one, 'student_model_', [metric], offset=0)

            kd_test_list.append(kd_batch_test_one_to_one)
            drca_test_list.append(drca_batch_test_one_to_one)
            kd_drca_test_list.append(kd_drca_batch_test_one_to_one)

        # 处理已有的预测新的每个批次，编号11到19 (CN comment)
        for n in range(2, 11): 
            kd_file_path_existing = os.path.join(kd_base_path_existing, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
            drca_file_path_existing = os.path.join(drca_base_path_existing, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_existing = os.path.join(kd_drca_base_path_existing, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')

            kd_batch_test_existing = process_data(kd_file_path_existing, 'student_model_', [metric], offset=9)
            drca_batch_test_existing = process_data(drca_file_path_existing, 'drca_model_', [metric], offset=9)
            kd_drca_batch_test_existing = process_data(kd_drca_file_path_existing, 'student_model_', [metric], offset=9)

            kd_test_list.append(kd_batch_test_existing)
            drca_test_list.append(drca_batch_test_existing)
            kd_drca_test_list.append(kd_drca_batch_test_existing)

        # 处理基线数据 (CN comment)
        baseline_one_to_one = process_data(baseline_paths[1], 'model_', [metric], offset=0)  # 一对一部分
        baseline_existing = process_data(baseline_paths[0], 'model_', [metric], offset=9)  # 已有的预测新的部分

        # 合并数据 (CN comment)
        data[metric] = {
            "Baseline": pd.concat([baseline_one_to_one, baseline_existing], ignore_index=True),
            "KD": pd.concat(kd_test_list, ignore_index=True),
            "DRCA": pd.concat(drca_test_list, ignore_index=True),
            "KD_DRCA": pd.concat(kd_drca_test_list, ignore_index=True)
        }
    return data

# 创建最终结果表格 (CN comment)
def create_results_table(metrics, metrics_name):
    data = process_all_data(metrics)
    output_path = "outputs/"
    
    for metric in metrics:
        table_data = []
        detailed_data = []

        for method in ['KD', 'DRCA', 'KD_DRCA']:
            baseline_data = data[metric]["Baseline"]
            method_data = data[metric][method]
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, results["positive"], results["neutral"], results["negative"], results["total"]])
            
            # 添加详细数据到detailed_data (CN comment)
            for model in range(2, 20):
                baseline_scores = baseline_data[baseline_data['Model'] == model][metric].values
                method_scores = method_data[method_data['Model'] == model][metric].values
                detailed_data.append([method, metric, model, list(baseline_scores), list(method_scores)])

        # 绘制表格 (CN comment)
        columns = ['Method', '+ (p<0.05)', '= (p>0.05)', '- (p<0.05)', 'Total']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.title(f'Significance Comparison for {metric} ({metrics_name})', fontsize=14)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison.png")
        plt.show()

        # 保存汇总数据为Excel文件 (CN comment)
        result_df = pd.DataFrame(table_data, columns=columns)
        result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_significance_comparison_data.xlsx", index=False)
        
        # 保存详细数据为Excel文件 (CN comment)
        detailed_columns = ['Method', 'Metric', 'Model', 'Baseline Scores', 'Method Scores']
        detailed_df = pd.DataFrame(detailed_data, columns=detailed_columns)
        detailed_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_detailed_data.xlsx", index=False)

# 调用函数生成图表 (CN comment)
create_results_table(test_metrics, 'Test Metrics')
create_results_table(val_metrics, 'Validation Metrics')

#看task1和task2各自的表现 (CN comment)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_base_path_existing = "outputs/"
drca_base_path_existing = "outputs/"
kd_drca_base_path_existing = "outputs/"

kd_base_path_one_to_one = "outputs/"
drca_base_path_one_to_one = "outputs/"
kd_drca_base_path_one_to_one = "outputs/"

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    for model in range(2, 20):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results["positive"] += 1
            else:
                results["negative"] += 1
        else:
            results["neutral"] += 1
        
        results["total"] += 1
    return results

# 处理所有数据 (CN comment)
# 处理所有数据，合并测试和验证数据 (CN comment)
def process_all_data(test_metrics, val_metrics):
    data = {}
    combined_metrics = [(test, val) for test, val in zip(test_metrics, val_metrics)]  # 合并测试和验证指标
    
    for test_metric, val_metric in combined_metrics:
        kd_test_list, drca_test_list, kd_drca_test_list = [], [], []

        # 处理一对一的每个批次，编号保持2～10不变 (CN comment)
        for n in range(2, 11):  # n的范围是2～10
            kd_file_path_one_to_one = os.path.join(kd_base_path_one_to_one, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
            drca_file_path_one_to_one = os.path.join(drca_base_path_one_to_one, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_one_to_one = os.path.join(kd_drca_base_path_one_to_one, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')

            # 处理数据，将测试和验证数据都读取并合并 (CN comment)
            kd_batch_test_one_to_one = process_data(kd_file_path_one_to_one, 'student_model_', [test_metric, val_metric], offset=0)
            drca_batch_test_one_to_one = process_data(drca_file_path_one_to_one, 'drca_model_', [test_metric, val_metric], offset=0)
            kd_drca_batch_test_one_to_one = process_data(kd_drca_file_path_one_to_one, 'student_model_', [test_metric, val_metric], offset=0)

            kd_test_list.append(kd_batch_test_one_to_one)
            drca_test_list.append(drca_batch_test_one_to_one)
            kd_drca_test_list.append(kd_drca_batch_test_one_to_one)

        # 处理已有的预测新的每个批次，编号11到19 (CN comment)
        for n in range(2, 11):
            kd_file_path_existing = os.path.join(kd_base_path_existing, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
            drca_file_path_existing = os.path.join(drca_base_path_existing, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_existing = os.path.join(kd_drca_base_path_existing, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')

            kd_batch_test_existing = process_data(kd_file_path_existing, 'student_model_', [test_metric, val_metric], offset=9)
            drca_batch_test_existing = process_data(drca_file_path_existing, 'drca_model_', [test_metric, val_metric], offset=9)
            kd_drca_batch_test_existing = process_data(kd_drca_file_path_existing, 'student_model_', [test_metric, val_metric], offset=9)

            kd_test_list.append(kd_batch_test_existing)
            drca_test_list.append(drca_batch_test_existing)
            kd_drca_test_list.append(kd_drca_batch_test_existing)

        # 处理基线数据 (CN comment)
        baseline_one_to_one = process_data(baseline_paths[1], 'model_', [test_metric, val_metric], offset=0)  # 一对一部分
        baseline_existing = process_data(baseline_paths[0], 'model_', [test_metric, val_metric], offset=9)  # 已有的预测新的部分

        # 合并数据 (CN comment)
        data[test_metric] = {
            "Baseline": pd.concat([baseline_one_to_one, baseline_existing], ignore_index=True),
            "KD": pd.concat(kd_test_list, ignore_index=True),
            "DRCA": pd.concat(drca_test_list, ignore_index=True),
            "KD_DRCA": pd.concat(kd_drca_test_list, ignore_index=True)
        }
    return data
# 处理Task 1和Task 2的分离 (CN comment)
def process_task_data(data, task_num):
    if task_num == 1:
        return data[data['Model'].between(2, 10)]
    elif task_num == 2:
        return data[data['Model'].between(11, 19)]


# 绘制图表 (CN comment)
def create_results_table(metrics, task_num, metrics_name):
    data = process_all_data(test_metrics, val_metrics)
    output_path = "outputs/"

    for metric in metrics:
        table_data = []
        detailed_data = []

        # 获取 Task 1 或 Task 2 的数据 (CN comment)
        baseline_data = process_task_data(data[metric]["Baseline"], task_num)
        kd_data = process_task_data(data[metric]["KD"], task_num)
        drca_data = process_task_data(data[metric]["DRCA"], task_num)
        kd_drca_data = process_task_data(data[metric]["KD_DRCA"], task_num)

        for method, method_data in [('KD', kd_data), ('DRCA', drca_data), ('KD_DRCA', kd_drca_data)]:
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, results["positive"], results["neutral"], results["negative"], results["total"]])

            for model in range(2, 20):
                baseline_scores = baseline_data[baseline_data['Model'] == model][metric].values
                method_scores = method_data[method_data['Model'] == model][metric].values
                detailed_data.append([method, metric, model, list(baseline_scores), list(method_scores)])

        # 保存结果并绘制图表 (CN comment)
        columns = ['Method', '+ (p<0.05)', '= (p>0.05)', '- (p<0.05)', 'Total']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.title(f'Significance Comparison for {metric} Task {task_num} ({metrics_name})', fontsize=14)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_significance_comparison.png")
        plt.show()

        # 保存汇总数据为Excel文件 (CN comment)
        result_df = pd.DataFrame(table_data, columns=columns)
        result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_significance_comparison_data.xlsx", index=False)

        # 保存详细数据为Excel文件 (CN comment)
        detailed_columns = ['Method', 'Metric', 'Model', 'Baseline Scores', 'Method Scores']
        detailed_df = pd.DataFrame(detailed_data, columns=detailed_columns)
        detailed_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_detailed_data.xlsx", index=False)


# 调用函数生成图表，为Task 1和Task 2分别生成图表 (CN comment)
create_results_table(test_metrics, 1, 'Metrics')  # 任务1
create_results_table(test_metrics, 2, 'Metrics')  # 任务2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_base_path_existing = "outputs/"
drca_base_path_existing = "outputs/"
kd_drca_base_path_existing = "outputs/"

kd_base_path_one_to_one = "outputs/"
drca_base_path_one_to_one = "outputs/"
kd_drca_base_path_one_to_one = "outputs/"

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据，将测试和验证数据都读取并合并 (CN comment)
def process_data(file_path, model_prefix, test_metric, val_metric, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据

    # 处理 test 和 val 两个指标 (CN comment)
    for metric in [test_metric, val_metric]:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    
    # 合并 test 和 val 的列 (CN comment)
    df['Combined Metric'] = (df[test_metric] + df[val_metric]) / 2.0
    return df[['Model', 'Combined Metric']]

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    for model in range(2, 20):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results["positive"] += 1
            else:
                results["negative"] += 1
        else:
            results["neutral"] += 1
        
        results["total"] += 1
    return results

# 处理所有数据，合并测试和验证数据 (CN comment)
def process_all_data(test_metrics, val_metrics):
    data = {}
    combined_metrics = [(test, val) for test, val in zip(test_metrics, val_metrics)]  # 合并测试和验证指标
    
    for test_metric, val_metric in combined_metrics:
        kd_test_list, drca_test_list, kd_drca_test_list = [], [], []

        # 处理一对一的每个批次，编号保持2～10不变 (CN comment)
        for n in range(2, 11):  # n的范围是2～10
            kd_file_path_one_to_one = os.path.join(kd_base_path_one_to_one, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
            drca_file_path_one_to_one = os.path.join(drca_base_path_one_to_one, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_one_to_one = os.path.join(kd_drca_base_path_one_to_one, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')

            # 调用process_data处理并合并测试和验证数据 (CN comment)
            kd_batch_test_one_to_one = process_data(kd_file_path_one_to_one, 'student_model_', test_metric, val_metric, offset=0)
            drca_batch_test_one_to_one = process_data(drca_file_path_one_to_one, 'drca_model_', test_metric, val_metric, offset=0)
            kd_drca_batch_test_one_to_one = process_data(kd_drca_file_path_one_to_one, 'student_model_', test_metric, val_metric, offset=0)

            kd_test_list.append(kd_batch_test_one_to_one)
            drca_test_list.append(drca_batch_test_one_to_one)
            kd_drca_test_list.append(kd_drca_batch_test_one_to_one)

        # 处理已有的预测新的每个批次，编号11到19 (CN comment)
        for n in range(2, 11):
            kd_file_path_existing = os.path.join(kd_base_path_existing, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
            drca_file_path_existing = os.path.join(drca_base_path_existing, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_existing = os.path.join(kd_drca_base_path_existing, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')

            # 调用process_data处理并合并测试和验证数据 (CN comment)
            kd_batch_test_existing = process_data(kd_file_path_existing, 'student_model_', test_metric, val_metric, offset=9)
            drca_batch_test_existing = process_data(drca_file_path_existing, 'drca_model_', test_metric, val_metric, offset=9)
            kd_drca_batch_test_existing = process_data(kd_drca_file_path_existing, 'student_model_', test_metric, val_metric, offset=9)

            kd_test_list.append(kd_batch_test_existing)
            drca_test_list.append(drca_batch_test_existing)
            kd_drca_test_list.append(kd_drca_batch_test_existing)

        # 处理基线数据 (CN comment)
        baseline_one_to_one = process_data(baseline_paths[1], 'model_', test_metric, val_metric, offset=0)  # 一对一部分
        baseline_existing = process_data(baseline_paths[0], 'model_', test_metric, val_metric, offset=9)  # 已有的预测新的部分

        # 合并数据 (CN comment)
        data[test_metric] = {
            "Baseline": pd.concat([baseline_one_to_one, baseline_existing], ignore_index=True),
            "KD": pd.concat(kd_test_list, ignore_index=True),
            "DRCA": pd.concat(drca_test_list, ignore_index=True),
            "KD_DRCA": pd.concat(kd_drca_test_list, ignore_index=True)
        }
    return data

# 处理Task 1和Task 2的分离 (CN comment)
def process_task_data(data, task_num):
    if task_num == 1:
        return data[data['Model'].between(2, 10)]
    elif task_num == 2:
        return data[data['Model'].between(11, 19)]

# 绘制图表 (CN comment)
def create_results_table(metrics, task_num, metrics_name):
    data = process_all_data(test_metrics, val_metrics)
    output_path = "outputs/"

    for metric in metrics:
        table_data = []
        detailed_data = []

        # 获取 Task 1 或 Task 2 的数据 (CN comment)
        baseline_data = process_task_data(data[metric]["Baseline"], task_num)
        kd_data = process_task_data(data[metric]["KD"], task_num)
        drca_data = process_task_data(data[metric]["DRCA"], task_num)
        kd_drca_data = process_task_data(data[metric]["KD_DRCA"], task_num)

        for method, method_data in [('KD', kd_data), ('DRCA', drca_data), ('KD_DRCA', kd_drca_data)]:
            results = compare_significance(baseline_data, method_data, 'Combined Metric')
            table_data.append([method, results["positive"], results["neutral"], results["negative"], results["total"]])

            for model in range(2, 20):
                baseline_scores = baseline_data[baseline_data['Model'] == model]['Combined Metric'].values
                method_scores = method_data[method_data['Model'] == model]['Combined Metric'].values
                detailed_data.append([method, metric, model, list(baseline_scores), list(method_scores)])

        # 保存结果并绘制图表 (CN comment)
        columns = ['Method', '+ (p<0.05)', '= (p>0.05)', '- (p<0.05)', 'Total']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.title(f'Significance Comparison for {metric} Task {task_num} ({metrics_name})', fontsize=14)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_significance_comparison.png")
        plt.show()

        # 保存汇总数据为Excel文件 (CN comment)
        result_df = pd.DataFrame(table_data, columns=columns)
        result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_significance_comparison_data.xlsx", index=False)

        # 保存详细数据为Excel文件 (CN comment)
        detailed_columns = ['Method', 'Metric', 'Model', 'Baseline Scores', 'Method Scores']
        detailed_df = pd.DataFrame(detailed_data, columns=detailed_columns)
        detailed_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_detailed_data.xlsx", index=False)


# 调用函数生成图表，为Task 1和Task 2分别生成图表 (CN comment)
create_results_table(test_metrics, 1, 'Metrics')  # 任务1
create_results_table(test_metrics, 2, 'Metrics')  # 任务2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_base_path_existing = "outputs/"
drca_base_path_existing = "outputs/"
kd_drca_base_path_existing = "outputs/"

kd_base_path_one_to_one = "outputs/"
drca_base_path_one_to_one = "outputs/"
kd_drca_base_path_one_to_one = "outputs/"

test_metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']
val_metrics = ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision']

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    for model in range(2, 20):
        baseline_scores = baseline[baseline['Model'] == model][metric].values
        method_scores = method_data[method_data['Model'] == model][metric].values
        
        if len(baseline_scores) == 0 or len(method_scores) == 0:
            continue
        
        baseline_scores = np.array(baseline_scores, dtype=float)
        method_scores = np.array(method_scores, dtype=float)
        
        t_stat, p_val = ttest_ind(baseline_scores, method_scores)
        mean_diff = method_scores.mean() - baseline_scores.mean()
        
        if p_val < 0.05:
            if mean_diff > 0:
                results["positive"] += 1
            else:
                results["negative"] += 1
        else:
            results["neutral"] += 1
        
        results["total"] += 1
    return results

# 处理所有数据，分别处理测试和验证数据 (CN comment)
def process_all_data(metrics):
    data = {}
    
    for metric in metrics:
        kd_test_list, drca_test_list, kd_drca_test_list = [], [], []

        # 处理一对一的每个批次，编号保持2～10不变 (CN comment)
        for n in range(2, 11):  # n的范围是2～10
            kd_file_path_one_to_one = os.path.join(kd_base_path_one_to_one, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
            drca_file_path_one_to_one = os.path.join(drca_base_path_one_to_one, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_one_to_one = os.path.join(kd_drca_base_path_one_to_one, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')

            # 处理数据，将测试或验证数据读取 (CN comment)
            kd_batch_test_one_to_one = process_data(kd_file_path_one_to_one, 'student_model_', [metric], offset=0)
            drca_batch_test_one_to_one = process_data(drca_file_path_one_to_one, 'drca_model_', [metric], offset=0)
            kd_drca_batch_test_one_to_one = process_data(kd_drca_file_path_one_to_one, 'student_model_', [metric], offset=0)

            kd_test_list.append(kd_batch_test_one_to_one)
            drca_test_list.append(drca_batch_test_one_to_one)
            kd_drca_test_list.append(kd_drca_batch_test_one_to_one)

        # 处理已有的预测新的每个批次，编号11到19 (CN comment)
        for n in range(2, 11):
            kd_file_path_existing = os.path.join(kd_base_path_existing, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
            drca_file_path_existing = os.path.join(drca_base_path_existing, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_existing = os.path.join(kd_drca_base_path_existing, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')

            kd_batch_test_existing = process_data(kd_file_path_existing, 'student_model_', [metric], offset=9)
            drca_batch_test_existing = process_data(drca_file_path_existing, 'drca_model_', [metric], offset=9)
            kd_drca_batch_test_existing = process_data(kd_drca_file_path_existing, 'student_model_', [metric], offset=9)

            kd_test_list.append(kd_batch_test_existing)
            drca_test_list.append(drca_batch_test_existing)
            kd_drca_test_list.append(kd_drca_batch_test_existing)

        # 处理基线数据 (CN comment)
        baseline_one_to_one = process_data(baseline_paths[1], 'model_', [metric], offset=0)  # 一对一部分
        baseline_existing = process_data(baseline_paths[0], 'model_', [metric], offset=9)  # 已有的预测新的部分

        # 合并数据 (CN comment)
        data[metric] = {
            "Baseline": pd.concat([baseline_one_to_one, baseline_existing], ignore_index=True),
            "KD": pd.concat(kd_test_list, ignore_index=True),
            "DRCA": pd.concat(drca_test_list, ignore_index=True),
            "KD_DRCA": pd.concat(kd_drca_test_list, ignore_index=True)
        }
    return data

# 处理Task 1和Task 2的分离 (CN comment)
def process_task_data(data, task_num):
    if task_num == 1:
        return data[data['Model'].between(2, 10)]
    elif task_num == 2:
        return data[data['Model'].between(11, 19)]

# 绘制图表 (CN comment)
def create_results_table(metrics, task_num, metrics_name, dataset_type):
    data = process_all_data(metrics)
    output_path = "outputs/"

    for metric in metrics:
        table_data = []
        detailed_data = []

        # 获取 Task 1 或 Task 2 的数据 (CN comment)
        baseline_data = process_task_data(data[metric]["Baseline"], task_num)
        kd_data = process_task_data(data[metric]["KD"], task_num)
        drca_data = process_task_data(data[metric]["DRCA"], task_num)
        kd_drca_data = process_task_data(data[metric]["KD_DRCA"], task_num)

        for method, method_data in [('KD', kd_data), ('DRCA', drca_data), ('KD_DRCA', kd_drca_data)]:
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, results["positive"], results["neutral"], results["negative"], results["total"]])

            for model in range(2, 20):
                baseline_scores = baseline_data[baseline_data['Model'] == model][metric].values
                method_scores = method_data[method_data['Model'] == model][metric].values
                detailed_data.append([method, metric, model, list(baseline_scores), list(method_scores)])

        # 保存结果并绘制图表 (CN comment)
        columns = ['Method', '+ (p<0.05)', '= (p>0.05)', '- (p<0.05)', 'Total']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.title(f'Significance Comparison for {metric} Task {task_num} ({metrics_name} - {dataset_type})', fontsize=14)
        plt.savefig(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_{dataset_type}_significance_comparison.png")
        plt.show()

    # 保存汇总数据为Excel文件 (CN comment)
    result_df = pd.DataFrame(table_data, columns=columns)
    result_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_{dataset_type}_significance_comparison_data.xlsx", index=False)

    # 保存详细数据为Excel文件 (CN comment)
    detailed_columns = ['Method', 'Metric', 'Model', 'Baseline Scores', 'Method Scores']
    detailed_df = pd.DataFrame(detailed_data, columns=detailed_columns)
    detailed_df.to_excel(output_path + f"{metric.replace(' ', '_').lower()}_task_{task_num}_{dataset_type}_detailed_data.xlsx", index=False)
create_results_table(test_metrics, 1, 'Metrics', 'Test') # Task 1 测试数据 
create_results_table(val_metrics, 1, 'Metrics', 'Validation') # Task 1 验证数据
create_results_table(test_metrics, 2, 'Metrics', 'Test') # Task 2 测试数据 
create_results_table(val_metrics, 2, 'Metrics', 'Validation') # Task 2 验证数据

import pandas as pd

# 示例Output file path（替换为实际路径） (CN comment)
file_path = "outputs/知识蒸馏一对一详细_batch_2_30遍.xlsx"

# 读取文件 (CN comment)
df = pd.read_excel(file_path)

# 打印列名 (CN comment)
print(df.columns)

#8.25放射图 (CN comment)

#所有最终数据读取总结 (CN comment)

import os
import pandas as pd
import numpy as np

# Output file path
baseline_paths = [
    "outputs/已有的预测新的详细.xlsx",
    "outputs/一对一详细.xlsx"
]

kd_base_path_existing = "outputs/"
drca_base_path_existing = "outputs/"
kd_drca_base_path_existing = "outputs/"

kd_base_path_one_to_one = "outputs/"
drca_base_path_one_to_one = "outputs/"
kd_drca_base_path_one_to_one = "outputs/"

# 读取并处理数据 (CN comment)
def process_data(file_path, model_prefix, metrics, offset=0):
    df = pd.read_excel(file_path)
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int) + offset
    df = df[df['Model'].between(2, 19)]  # 只处理模型2到19的数据
    for metric in metrics:
        df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
        df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
        df = df.explode(metric)
        df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# 处理所有数据 (CN comment)
def process_all_data(metrics):
    data = {}
    for metric in metrics:
        kd_test_list, drca_test_list, kd_drca_test_list = [], [], []

        # 处理一对一的每个批次，编号保持2～10不变 (CN comment)
        for n in range(2, 11):  # n的范围是2～10
            kd_file_path_one_to_one = os.path.join(kd_base_path_one_to_one, f'知识蒸馏一对一详细_batch_{n}_30遍.xlsx')
            drca_file_path_one_to_one = os.path.join(drca_base_path_one_to_one, f'DRCA一对一详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_one_to_one = os.path.join(kd_drca_base_path_one_to_one, f'DRCA_KD一对一详细_batch_{n}_30遍.xlsx')

            kd_batch_test_one_to_one = process_data(kd_file_path_one_to_one, 'student_model_', [metric], offset=0)
            drca_batch_test_one_to_one = process_data(drca_file_path_one_to_one, 'drca_model_', [metric], offset=0)
            kd_drca_batch_test_one_to_one = process_data(kd_drca_file_path_one_to_one, 'student_model_', [metric], offset=0)

            kd_test_list.append(kd_batch_test_one_to_one)
            drca_test_list.append(drca_batch_test_one_to_one)
            kd_drca_test_list.append(kd_drca_batch_test_one_to_one)

        # 处理已有的预测新的每个批次，编号11到19 (CN comment)
        for n in range(2, 11): 
            kd_file_path_existing = os.path.join(kd_base_path_existing, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
            drca_file_path_existing = os.path.join(drca_base_path_existing, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
            kd_drca_file_path_existing = os.path.join(kd_drca_base_path_existing, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')

            kd_batch_test_existing = process_data(kd_file_path_existing, 'student_model_', [metric], offset=9)
            drca_batch_test_existing = process_data(drca_file_path_existing, 'drca_model_', [metric], offset=9)
            kd_drca_batch_test_existing = process_data(kd_drca_file_path_existing, 'student_model_', [metric], offset=9)

            kd_test_list.append(kd_batch_test_existing)
            drca_test_list.append(drca_batch_test_existing)
            kd_drca_test_list.append(kd_drca_batch_test_existing)

        # 处理基线数据 (CN comment)
        baseline_one_to_one = process_data(baseline_paths[1], 'model_', [metric], offset=0)  # 一对一部分
        baseline_existing = process_data(baseline_paths[0], 'model_', [metric], offset=9)  # 已有的预测新的部分

        # 合并数据 (CN comment)
        data[metric] = {
            "Baseline": pd.concat([baseline_one_to_one, baseline_existing], ignore_index=True),
            "KD": pd.concat(kd_test_list, ignore_index=True),
            "DRCA": pd.concat(drca_test_list, ignore_index=True),
            "KD_DRCA": pd.concat(kd_drca_test_list, ignore_index=True)
        }
    return data

# 调用函数读取并处理测试数据 (CN comment)
test_data = process_all_data(['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision'])

# 调用函数读取并处理验证数据 (CN comment)
val_data = process_all_data(['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision'])

pip install xlsxwriter

#雷达图绘制中位置 (CN comment)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # 引入正则表达式库

# 处理好的数据 (CN comment)
test_data = process_all_data(['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision'])
val_data = process_all_data(['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision'])

# 保存数据和绘制雷达图的函数 (CN comment)
def save_and_plot_radar_charts(data, metrics, metrics_name, output_path):
    categories = [f"Task {i} Batch {j}" for i in range(1, 3) for j in range(2, 11)]
    
    # 创建保存数据的 Excel 文件 (CN comment)
    writer = pd.ExcelWriter(os.path.join(output_path, f'{metrics_name}_median_data.xlsx'), engine='xlsxwriter')
    
    for metric in metrics:
        baseline_medians, kd_medians, drca_medians, kd_drca_medians = [], [], [], []
        
        for cat in categories:
            # 使用正则表达式提取task和batch的数字 (CN comment)
            task_num = int(re.search(r'Task (\d+)', cat).group(1))
            batch_num = int(re.search(r'Batch (\d+)', cat).group(1))
            
            if task_num == 1:  # Task 1 对应一对一
                baseline_medians.append(data[metric]['Baseline'][(data[metric]['Baseline']['Model'] == batch_num)][metric].median())
                kd_medians.append(data[metric]['KD'][(data[metric]['KD']['Model'] == batch_num)][metric].median())
                drca_medians.append(data[metric]['DRCA'][(data[metric]['DRCA']['Model'] == batch_num)][metric].median())
                kd_drca_medians.append(data[metric]['KD_DRCA'][(data[metric]['KD_DRCA']['Model'] == batch_num)][metric].median())
            else:  # Task 2 对应已有的预测新的
                batch_num += 9
                baseline_medians.append(data[metric]['Baseline'][(data[metric]['Baseline']['Model'] == batch_num)][metric].median())
                kd_medians.append(data[metric]['KD'][(data[metric]['KD']['Model'] == batch_num)][metric].median())
                drca_medians.append(data[metric]['DRCA'][(data[metric]['DRCA']['Model'] == batch_num)][metric].median())
                kd_drca_medians.append(data[metric]['KD_DRCA'][(data[metric]['KD_DRCA']['Model'] == batch_num)][metric].median())

        # 归一化处理，以Baseline为基准 (CN comment)
        baseline_medians = np.array(baseline_medians)
        kd_medians = np.array(kd_medians) / baseline_medians
        drca_medians = np.array(drca_medians) / baseline_medians
        kd_drca_medians = np.array(kd_drca_medians) / baseline_medians
        baseline_medians = np.ones_like(baseline_medians)  # 基线值归一化为1

        # 保存数据到Excel (CN comment)
        df = pd.DataFrame({
            'Category': categories,
            'Baseline': baseline_medians,
            'KD': kd_medians,
            'DRCA': drca_medians,
            'KD-DRCA': kd_drca_medians
        })
        df.to_excel(writer, sheet_name=metric.replace(' ', '_'), index=False)

        # 绘制雷达图 (CN comment)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(polar=True))
        
        # 绘制每种方法的雷达图，线条加粗 (CN comment)
        ax.plot(angles, np.concatenate([baseline_medians, baseline_medians[:1]]), linewidth=4, linestyle='solid', label='Baseline')
        ax.plot(angles, np.concatenate([kd_medians, kd_medians[:1]]), linewidth=4, linestyle='dashed', label='KD')
        ax.plot(angles, np.concatenate([drca_medians, drca_medians[:1]]), linewidth=4, linestyle='dotted', label='DRCA')
        ax.plot(angles, np.concatenate([kd_drca_medians, kd_drca_medians[:1]]), linewidth=4, linestyle='dashdot', label='KD-DRCA')

        # 设置图例和标签 (CN comment)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, fontsize=20)
        ax.set_rlabel_position(0)
        plt.yticks([0.8, 1.0, 1.3], ["0.8", "1.0", "1.3"], color="grey", size=15)
        plt.ylim(0.8, 1.3)

        plt.title(f'{metric}', size=40, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), prop={'size': 20})  # 设置图例及其文字的大小

        # 保存图像 (CN comment)
        plt.savefig(os.path.join(output_path, f'{metric.replace(" ", "_").lower()}_radar_chart.png'), bbox_inches='tight')
        plt.close()

    writer.close()

# 输出路径 (CN comment)
output_path = "outputs/"

# 保存并绘制测试数据的雷达图 (CN comment)
save_and_plot_radar_charts(test_data, ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision'], 'Test Metrics', output_path)

# 保存并绘制验证数据的雷达图 (CN comment)
save_and_plot_radar_charts(val_data, ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision'], 'Validation Metrics', output_path)

#雷达图绘制平均值 (CN comment)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # 引入正则表达式库

# 处理好的数据 (CN comment)
test_data = process_all_data(['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision'])
val_data = process_all_data(['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision'])

# 保存数据和绘制雷达图的函数 (CN comment)
def save_and_plot_radar_charts(data, metrics, metrics_name, output_path):
    categories = [f"Task {i} Batch {j}" for i in range(1, 3) for j in range(2, 11)]
    
    # 创建保存数据的 Excel 文件 (CN comment)
    writer = pd.ExcelWriter(os.path.join(output_path, f'{metrics_name}_mean_data.xlsx'), engine='xlsxwriter')
    
    for metric in metrics:
        baseline_means, kd_means, drca_means, kd_drca_means = [], [], [], []
        
        for cat in categories:
            # 使用正则表达式提取task和batch的数字 (CN comment)
            task_num = int(re.search(r'Task (\d+)', cat).group(1))
            batch_num = int(re.search(r'Batch (\d+)', cat).group(1))
            
            if task_num == 1:  # Task 1 对应一对一
                baseline_means.append(data[metric]['Baseline'][(data[metric]['Baseline']['Model'] == batch_num)][metric].mean())
                kd_means.append(data[metric]['KD'][(data[metric]['KD']['Model'] == batch_num)][metric].mean())
                drca_means.append(data[metric]['DRCA'][(data[metric]['DRCA']['Model'] == batch_num)][metric].mean())
                kd_drca_means.append(data[metric]['KD_DRCA'][(data[metric]['KD_DRCA']['Model'] == batch_num)][metric].mean())
            else:  # Task 2 对应已有的预测新的
                batch_num += 9
                baseline_means.append(data[metric]['Baseline'][(data[metric]['Baseline']['Model'] == batch_num)][metric].mean())
                kd_means.append(data[metric]['KD'][(data[metric]['KD']['Model'] == batch_num)][metric].mean())
                drca_means.append(data[metric]['DRCA'][(data[metric]['DRCA']['Model'] == batch_num)][metric].mean())
                kd_drca_means.append(data[metric]['KD_DRCA'][(data[metric]['KD_DRCA']['Model'] == batch_num)][metric].mean())

        # 归一化处理，以Baseline为基准 (CN comment)
        baseline_means = np.array(baseline_means)
        kd_means = np.array(kd_means) / baseline_means
        drca_means = np.array(drca_means) / baseline_means
        kd_drca_means = np.array(kd_drca_means) / baseline_means
        baseline_means = np.ones_like(baseline_means)  # 基线值归一化为1

        # 保存数据到Excel (CN comment)
        df = pd.DataFrame({
            'Category': categories,
            'Baseline': baseline_means,
            'KD': kd_means,
            'DRCA': drca_means,
            'KD-DRCA': kd_drca_means
        })
        df.to_excel(writer, sheet_name=metric.replace(' ', '_'), index=False)

        # 绘制雷达图 (CN comment)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(polar=True))
        
        # 绘制每种方法的雷达图，线条加粗 (CN comment)
        ax.plot(angles, np.concatenate([baseline_means, baseline_means[:1]]), linewidth=4, linestyle='solid', label='Baseline')
        ax.plot(angles, np.concatenate([kd_means, kd_means[:1]]), linewidth=4, linestyle='dashed', label='KD')
        ax.plot(angles, np.concatenate([drca_means, drca_means[:1]]), linewidth=4, linestyle='dotted', label='DRCA')
        ax.plot(angles, np.concatenate([kd_drca_means, kd_drca_means[:1]]), linewidth=4, linestyle='dashdot', label='KD-DRCA')

        # 设置图例和标签 (CN comment)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, fontsize=20)
        ax.set_rlabel_position(0)
        plt.yticks([0.8, 1.0, 1.3], ["0.8", "1.0", "1.3"], color="grey", size=15)
        plt.ylim(0.8, 1.3)

        plt.title(f'{metric}', size=40, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), prop={'size': 20})  # 设置图例及其文字的大小

        # 保存图像 (CN comment)
        plt.savefig(os.path.join(output_path, f'{metric.replace(" ", "_").lower()}_mean_radar_chart.png'), bbox_inches='tight')
        plt.close()

    writer.close()

# 输出路径 (CN comment)
output_path = "outputs/"

# 保存并绘制测试数据的雷达图 (CN comment)
save_and_plot_radar_charts(test_data, ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision'], 'Test Metrics', output_path)

# 保存并绘制验证数据的雷达图 (CN comment)
save_and_plot_radar_charts(val_data, ['Val Accuracy', 'Val F1 Score', 'Val Recall', 'Val Precision'], 'Validation Metrics', output_path)

#放射图 拼接 (CN comment)

from PIL import Image

def combine_images():
    # 保存拼接图的路径 (CN comment)
    save_path_combined = "outputs/"
    if not os.path.exists(save_path_combined):
        os.makedirs(save_path_combined)
    
    # 中位数图像Output file path (CN comment)
    median_path = "outputs/"
    # 平均值图像Output file path (CN comment)
    avg_path = "outputs/"
    
    # 获取中位数和平均值图像文件名（确保按顺序排列） (CN comment)
    median_images = sorted([f for f in os.listdir(median_path) if f.endswith("_radar_chart_median.png")])
    avg_images = sorted([f for f in os.listdir(avg_path) if f.endswith("_radar_chart_mean.png")])
    
    # 打印文件名列表以检查是否有文件缺失或命名不一致 (CN comment)
    print("Median Images:", median_images)
    print("Average Images:", avg_images)
    
    # 确保两个目录下的文件数量和顺序匹配 (CN comment)
    assert len(median_images) == len(avg_images), "图像数量不匹配，请检查生成的图像文件。"
    
    # 创建一个新画布，宽度为两张图的宽度，长度为单列图像数乘以图像高度 (CN comment)
    img_sample = Image.open(os.path.join(median_path, median_images[0]))
    img_width, img_height = img_sample.size
    total_width = img_width * 2  # 两列
    total_height = img_height * len(median_images)  # 8行
    
    # 创建空白画布 (CN comment)
    combined_image = Image.new('RGB', (total_width, total_height))
    
    # 依次将中位数和平均值图像拼接到画布上 (CN comment)
    for i in range(len(median_images)):
        median_img = Image.open(os.path.join(median_path, median_images[i]))
        avg_img = Image.open(os.path.join(avg_path, avg_images[i]))
        
        # 将中位数图像粘贴到左列 (CN comment)
        combined_image.paste(median_img, (0, i * img_height))
        # 将平均值图像粘贴到右列 (CN comment)
        combined_image.paste(avg_img, (img_width, i * img_height))
    
    # 保存最终拼接的图像 (CN comment)
    combined_image.save(os.path.join(save_path_combined, "combined_radar_charts.png"))
    combined_image.show()

# 调用函数拼接图像 (CN comment)
combine_images()

import os
import matplotlib.pyplot as plt
from PIL import Image

def combine_task1_images():
    # 定义图像路径 (CN comment)
    image_paths = [
        "outputs/accuracy_test.png",
        "outputs/accuracy_val.png",
        "outputs/f1_score_test.png",
        "outputs/f1_score_val.png",
        "outputs/precision_test.png",
        "outputs/precision_val.png",
        "outputs/recall_test.png",
        "outputs/recall_val.png"
    ]
    
    # 打开所有图像并确定它们的大小 (CN comment)
    images = [Image.open(image) for image in image_paths]
    widths, heights = zip(*(img.size for img in images))

    # 定义拼接后的图像尺寸 (CN comment)
    total_width = max(widths) * 2  # 两列
    total_height = max(heights) * 4  # 四行

    # 创建空白画布 (CN comment)
    combined_image = Image.new('RGB', (total_width, total_height))

    # 依次将图像粘贴到画布上 (CN comment)
    for i, img in enumerate(images):
        x_offset = (i % 2) * max(widths)  # 左/右列
        y_offset = (i // 2) * max(heights)  # 上下行
        combined_image.paste(img, (x_offset, y_offset))
    
    # 保存拼接后的图像 (CN comment)
    combined_save_path = "outputs/combined_task1.png"
    combined_image.save(combined_save_path)
    combined_image.show()

# 调用函数拼接图像 (CN comment)
combine_task1_images()

import os
import matplotlib.pyplot as plt
from PIL import Image

def combine_task2_images():
    # 定义图像路径 (CN comment)
    image_paths = [
        "outputs/accuracy_test.png",
        "outputs/accuracy_val.png",
        "outputs/f1_score_test.png",
        "outputs/f1_score_val.png",
        "outputs/precision_test.png",
        "outputs/precision_val.png",
        "outputs/recall_test.png",
        "outputs/recall_val.png"
    ]
    
    # 打开所有图像并确定它们的大小 (CN comment)
    images = [Image.open(image) for image in image_paths]
    widths, heights = zip(*(img.size for img in images))

    # 定义拼接后的图像尺寸 (CN comment)
    total_width = max(widths) * 2  # 两列
    total_height = max(heights) * 4  # 四行

    # 创建空白画布 (CN comment)
    combined_image = Image.new('RGB', (total_width, total_height))

    # 依次将图像粘贴到画布上 (CN comment)
    for i, img in enumerate(images):
        x_offset = (i % 2) * max(widths)  # 左/右列
        y_offset = (i // 2) * max(heights)  # 上下行
        combined_image.paste(img, (x_offset, y_offset))
    
    # 保存拼接后的图像 (CN comment)
    combined_save_path = "outputs/combined_task2.png"
    combined_image.save(combined_save_path)
    combined_image.show()

# 调用函数拼接图像 (CN comment)
combine_task2_images()

from PIL import Image, ImageOps

def combine_tsne_images_tightly():
    # Output file path列表 (CN comment)
    image_paths = [
        "outputs/tsne_visualization_batch_1.png",
        "outputs/tsne_visualization_batch_2.png",
        "outputs/tsne_visualization_batch_3.png",
        "outputs/tsne_visualization_batch_4.png",
        "outputs/tsne_visualization_batch_5.png",
        "outputs/tsne_visualization_batch_6.png",
        "outputs/tsne_visualization_batch_7.png",
        "outputs/tsne_visualization_batch_8.png",
        "outputs/tsne_visualization_batch_9.png",
        "outputs/tsne_visualization_batch_10.png"
    ]
    
    # 打开并裁剪所有图像的边距 (CN comment)
    images = [ImageOps.crop(Image.open(path), border=90) for path in image_paths]
    
    # 获取单张图像的宽度和高度 (CN comment)
    img_width, img_height = images[0].size
    
    # 设置新画布的大小：两列，每列 5 张图片 (CN comment)
    total_width = img_width * 2
    total_height = img_height * 5
    
    # 创建空白画布 (CN comment)
    combined_image = Image.new('RGB', (total_width, total_height))
    
    # 将图像依次粘贴到新画布上 (CN comment)
    for i, img in enumerate(images):
        x_offset = (i // 5) * img_width  # 列的偏移
        y_offset = (i % 5) * img_height  # 行的偏移
        combined_image.paste(img, (x_offset, y_offset))
    
    # 保存拼接后的图像 (CN comment)
    combined_image.save("outputs/tsne_visualization_combined.png")
    combined_image.show()

# 调用函数拼接图像 (CN comment)
combine_tsne_images_tightly()
