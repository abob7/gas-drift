#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reproducibility script (exported from notebook).

Place dataset files under data/raw/ before running.
"""

import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

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

#baseline 7.25
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

for j in range(2, 11):
    Xt, Yt = all_data[j-1]  # 当前批次用作测试
    X_train_combined = np.empty((0, Xt.shape[1]))
    Y_train_combined = np.empty((0, 6))
    
    # 合并前 j-1 个批次的数据用于训练 (CN comment)
    for k in range(1, j):
        X_train, Y_train = all_data[k-1]
        X_train_combined = np.vstack((X_train_combined, X_train))
        Y_train_combined = np.vstack((Y_train_combined, Y_train))
    
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
        model = create_model(X_train_combined.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_combined, Y_train_combined, epochs=10, batch_size=32, verbose=0)
        
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
detailed_save_path = "outputs/已有的预测新的详细.xlsx"
detailed_results_df.to_excel(detailed_save_path, index=False)

# 保存汇总结果到Excel文件 (CN comment)
summary_save_path = "outputs/已有的预测新的汇总.xlsx"
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

# Figure S1: KD temperature sensitivity analysis
# Representative task: Task 2, train on batches 1-9, predict batch 10
# Output dir: outputs/ (CN comment)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt

# -----------------------------
# 0) Config
# -----------------------------
OUT_DIR = "outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# Wider T range (as in your paper)
TEMPERATURE_GRID = [0.3, 1, 2, 3, 5, 25, 50, 100, 200]

N_REPEATS = 30
EPOCHS_TEACHER = 20
EPOCHS_STUDENT = 20
BATCH_SIZE = 32

# For reproducibility (not perfectly deterministic on GPU, but helps)
BASE_SEED = 20250727

# -----------------------------
# 1) Distillation loss (keep consistent with your current implementation)
# NOTE: This matches your pipeline; do NOT change if you want comparable results.
# -----------------------------
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return categorical_crossentropy(soft_true, soft_pred)

# -----------------------------
# 2) Load & normalize all batches together (same as you did)
# Requires: process_data(), prepare_data() already defined in your preamble.
# -----------------------------
all_data = []
for j in range(1, 11):
    data = process_data(f"data/raw/batch{j}.dat")
    X, Y = prepare_data(data)  # Y should be one-hot (N,6)
    all_data.append((X, Y))

X_combined = np.vstack([d[0] for d in all_data])
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

start = 0
for j in range(10):
    end = start + all_data[j][0].shape[0]
    all_data[j] = (X_combined_scaled[start:end], all_data[j][1])
    start = end

# -----------------------------
# 3) Build Task 2, n=10: train on batch1..9, target=batch10
# -----------------------------
X_train_combined = np.vstack([all_data[k][0] for k in range(0, 9)])  # batch1..9
Y_train_combined = np.vstack([all_data[k][1] for k in range(0, 9)])
Xt, Yt = all_data[9]  # batch10

# -----------------------------
# 4) Run T scan
# detailed: each repeat × each T
# summary: mean/std per T
# -----------------------------
detailed_rows = []
summary_rows = []

for T in TEMPERATURE_GRID:
    val_accs, val_f1s, val_precs, val_recs = [], [], [], []
    test_accs, test_f1s, test_precs, test_recs = [], [], [], []

    for r in range(1, N_REPEATS + 1):
        seed = BASE_SEED + r
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Split target batch10 into val/test
        X_val, X_test, Y_val, Y_test = train_test_split(
            Xt, Yt, test_size=0.5, random_state=r
        )

        # Create models (requires your create_model(input_dim))
        teacher_model = create_model(X_train_combined.shape[1])
        student_model = create_model(X_train_combined.shape[1])

        teacher_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        student_model.compile(
            optimizer="adam",
            loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, T),
            metrics=["accuracy"],
        )

        # Train teacher on batch1..9
        teacher_model.fit(
            X_train_combined, Y_train_combined,
            epochs=EPOCHS_TEACHER, batch_size=BATCH_SIZE, verbose=0
        )

        # Soft labels for train + val
        soft_train = teacher_model.predict(X_train_combined, verbose=0)
        soft_val = teacher_model.predict(X_val, verbose=0)

        # Student trains on (train + val) with teacher soft labels
        X_mix = np.vstack((X_train_combined, X_val))
        y_soft_mix = np.vstack((soft_train, soft_val))

        student_model.fit(
            X_mix, y_soft_mix,
            epochs=EPOCHS_STUDENT, batch_size=BATCH_SIZE, verbose=0
        )

        # ---- Evaluate on VAL (hard labels) ----
        _, val_acc = student_model.evaluate(X_val, Y_val, verbose=0)
        Y_val_pred = student_model.predict(X_val, verbose=0)
        Y_val_pred_cls = np.argmax(Y_val_pred, axis=1)
        Y_val_cls = np.argmax(Y_val, axis=1)

        val_f1 = f1_score(Y_val_cls, Y_val_pred_cls, average="macro", zero_division=1)
        val_prec = precision_score(Y_val_cls, Y_val_pred_cls, average="macro", zero_division=1)
        val_rec = recall_score(Y_val_cls, Y_val_pred_cls, average="macro", zero_division=1)

        # ---- Evaluate on TEST ----
        _, test_acc = student_model.evaluate(X_test, Y_test, verbose=0)
        Y_test_pred = student_model.predict(X_test, verbose=0)
        Y_test_pred_cls = np.argmax(Y_test_pred, axis=1)
        Y_test_cls = np.argmax(Y_test, axis=1)

        test_f1 = f1_score(Y_test_cls, Y_test_pred_cls, average="macro", zero_division=1)
        test_prec = precision_score(Y_test_cls, Y_test_pred_cls, average="macro", zero_division=1)
        test_rec = recall_score(Y_test_cls, Y_test_pred_cls, average="macro", zero_division=1)

        # Collect
        val_accs.append(val_acc); val_f1s.append(val_f1); val_precs.append(val_prec); val_recs.append(val_rec)
        test_accs.append(test_acc); test_f1s.append(test_f1); test_precs.append(test_prec); test_recs.append(test_rec)

        detailed_rows.append({
            "Task": "Task2_train1to9_test10",
            "Temperature": T,
            "Repeat": r,
            "Val Accuracy": val_acc,
            "Val F1": val_f1,
            "Val Precision": val_prec,
            "Val Recall": val_rec,
            "Test Accuracy": test_acc,
            "Test F1": test_f1,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "RandomState": r
        })

    # Summary for this T
    summary_rows.append({
        "Task": "Task2_train1to9_test10",
        "Temperature": T,

        "Val Accuracy mean": float(np.mean(val_accs)),
        "Val Accuracy std": float(np.std(val_accs, ddof=1)),
        "Val F1 mean": float(np.mean(val_f1s)),
        "Val F1 std": float(np.std(val_f1s, ddof=1)),
        "Val Precision mean": float(np.mean(val_precs)),
        "Val Precision std": float(np.std(val_precs, ddof=1)),
        "Val Recall mean": float(np.mean(val_recs)),
        "Val Recall std": float(np.std(val_recs, ddof=1)),

        "Test Accuracy mean": float(np.mean(test_accs)),
        "Test Accuracy std": float(np.std(test_accs, ddof=1)),
        "Test F1 mean": float(np.mean(test_f1s)),
        "Test F1 std": float(np.std(test_f1s, ddof=1)),
        "Test Precision mean": float(np.mean(test_precs)),
        "Test Precision std": float(np.std(test_precs, ddof=1)),
        "Test Recall mean": float(np.mean(test_recs)),
        "Test Recall std": float(np.std(test_recs, ddof=1)),
    })

# Save Excels
detailed_df = pd.DataFrame(detailed_rows)
summary_df = pd.DataFrame(summary_rows)

detailed_path = os.path.join(OUT_DIR, "T_scan_Task2_batch10_detailed.xlsx")
summary_path = os.path.join(OUT_DIR, "T_scan_Task2_batch10_summary.xlsx")

detailed_df.to_excel(detailed_path, index=False)
summary_df.to_excel(summary_path, index=False)

print("Saved detailed:", detailed_path)
print("Saved summary:", summary_path)

# -----------------------------
# 5) Plot mean±std (Figure S1)
# Suggest: show TEST metrics (Accuracy + Macro F1) with error bars
# -----------------------------
temps = summary_df["Temperature"].values.astype(float)

acc_mean = summary_df["Test Accuracy mean"].values
acc_std  = summary_df["Test Accuracy std"].values

f1_mean  = summary_df["Test F1 mean"].values
f1_std   = summary_df["Test F1 std"].values

# (A) Test Accuracy vs T
plt.figure(figsize=(7, 4.5))
plt.errorbar(temps, acc_mean, yerr=acc_std, marker="o", capsize=4)
plt.xscale("log")
plt.xlabel("Temperature T (log scale)")
plt.ylabel("Test Accuracy (mean ± std)")
plt.title("Figure S1A. KD temperature sensitivity (Task2: train batch1-9 → test batch10)")
plt.tight_layout()
figA_png = os.path.join(OUT_DIR, "Figure_S1A_T_sensitivity_TestAccuracy.png")
figA_pdf = os.path.join(OUT_DIR, "Figure_S1A_T_sensitivity_TestAccuracy.pdf")
plt.savefig(figA_png, dpi=300)
plt.savefig(figA_pdf)
plt.close()

# (B) Test Macro-F1 vs T
plt.figure(figsize=(7, 4.5))
plt.errorbar(temps, f1_mean, yerr=f1_std, marker="o", capsize=4)
plt.xscale("log")
plt.xlabel("Temperature T (log scale)")
plt.ylabel("Test Macro-F1 (mean ± std)")
plt.title("Figure S1B. KD temperature sensitivity (Task2: train batch1-9 → test batch10)")
plt.tight_layout()
figB_png = os.path.join(OUT_DIR, "Figure_S1B_T_sensitivity_TestMacroF1.png")
figB_pdf = os.path.join(OUT_DIR, "Figure_S1B_T_sensitivity_TestMacroF1.pdf")
plt.savefig(figB_png, dpi=300)
plt.savefig(figB_pdf)
plt.close()

print("Saved figures:")
print(figA_png)
print(figB_png)

# KD 7.27
# 用知识蒸馏给每个任务定制温度 (CN comment)
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

# Define candidate temperature values
temperature_options = [50, 100, 200]

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

for j in range(2, 11):
    Xt, Yt = all_data[j-1]  # 当前批次用作测试
    X_train_combined = np.empty((0, Xt.shape[1]))
    Y_train_combined = np.empty((0, 6))

    # 合并前 j-1 个批次的数据用于训练 (CN comment)
    for k in range(1, j):
        X_train, Y_train = all_data[k-1]
        X_train_combined = np.vstack((X_train_combined, X_train))
        Y_train_combined = np.vstack((Y_train_combined, Y_train))

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

        for experiment in range(1, 6):  # 进行5次实验
            # Split the target batch into validation and test sets
            X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

            # Initialize teacher/student models
            teacher_model = create_model(X_train_combined.shape[1])
            student_model = create_model(X_train_combined.shape[1])
            teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, temperature), metrics=['accuracy'])

            # Train the teacher model
            teacher_model.fit(X_train_combined, Y_train_combined, epochs=20, batch_size=32, verbose=0)

            # Generate soft labels for training and validation sets
            soft_labels_train = teacher_model.predict(X_train_combined)
            soft_labels_val = teacher_model.predict(X_val)

            # Merge soft labels
            soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
            X_combined_with_val = np.vstack((X_train_combined, X_val))

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
detailed_results_df.to_excel("outputs/知识蒸馏已有的预测新的（定制版）详细.xlsx", index=False)
summary_results_df.to_excel("outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx", index=False)

# Print summary results
print(summary_results_df)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return tf.keras.losses.categorical_crossentropy(soft_true, soft_pred)

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

# 读取最佳条件并进行30次重复实验 (CN comment)
summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
summary_df = pd.read_excel(summary_path)

for j in range(2, 11):
    best_temperature = summary_df.loc[summary_df['Model'] == f'student_model_{j}', 'Best Temperature'].values[0]

    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    X_train_combined = np.empty((0, Xt.shape[1]))
    Y_train_combined = np.empty((0, 6))

    # 合并前 j-1 个批次的数据用于训练 (CN comment)
    for k in range(1, j):
        X_train, Y_train = all_data[k-1]
        X_train_combined = np.vstack((X_train_combined, X_train))
        Y_train_combined = np.vstack((Y_train_combined, Y_train))

    for experiment in range(1, 31):  # 进行30次实验
        X_val, X_test, Y_val, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=experiment)

        # Initialize teacher/student models
        teacher_model = create_model(X_train_combined.shape[1])
        student_model = create_model(X_train_combined.shape[1])
        teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, best_temperature), metrics=['accuracy'])

        # Train the teacher model
        teacher_model.fit(X_train_combined, Y_train_combined, epochs=20, batch_size=32, verbose=0)

        # Generate soft labels for training and validation sets
        soft_labels_train = teacher_model.predict(X_train_combined)
        soft_labels_val = teacher_model.predict(X_val)

        # Merge soft labels
        soft_labels_combined = np.vstack((soft_labels_train, soft_labels_val))
        X_combined_with_val = np.vstack((X_train_combined, X_val))

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
    detailed_save_path = f"outputs/知识蒸馏已有的预测新的详细_batch_{j}_30遍.xlsx"
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

# DRCA 7.27
# 定制DRCA (CN comment)
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
n_components_options = [5, 10, 15, 20, 30, 40]
alpha_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Initialize result containers
results = []
detailed_accuracies = []

# 修改任务范围 (CN comment)
for j in range(2, 11):
    Xt, Yt = all_data[j-1]
    Xs = np.vstack([all_data[k-1][0] for k in range(1, j)])  # 合并前j-1个批次的数据
    Ys = np.vstack([all_data[k-1][1] for k in range(1, j)])

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

            for experiment in range(1, 6):  # 进行5次实验
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
detailed_save_path = "outputs/DRCA已有的预测新的（定制版）详细.xlsx"
detailed_results_df.to_excel(detailed_save_path, index=False)

# 保存汇总结果到Excel文件 (CN comment)
summary_save_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
summary_results_df.to_excel(summary_save_path, index=False)

# Print summary results进行检查 (CN comment)
print(summary_results_df)

#30
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
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

# 读取最佳条件并进行30次重复实验 (CN comment)
summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
summary_df = pd.read_excel(summary_path)

for j in range(2, 11):
    best_n_components = summary_df.loc[summary_df['Model'] == f'drca_model_{j}', 'Best n_components'].values[0]
    best_alpha = summary_df.loc[summary_df['Model'] == f'drca_model_{j}', 'Best Alpha'].values[0]

    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs = np.vstack([all_data[k-1][0] for k in range(1, j)])  # 合并前j-1个批次的数据
    Ys = np.vstack([all_data[k-1][1] for k in range(1, j)])

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
    detailed_save_path = f"outputs/DRCA已有的预测新的详细_batch_{j}_30遍.xlsx"
    detailed_results_df.to_excel(detailed_save_path, index=False)

    # 打印完成信息 (CN comment)
    print(f"Batch {j} - 30 repetitions completed and saved to {detailed_save_path}")

#KD和DRCA结合 (CN comment)

# KD和DRCA结合 7.28（对温度也进行了定制而不是读取单独KD模型里的最佳温度） (CN comment)
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

results = []
detailed_accuracies = []

# 对批次2到批次10进行预测并记录结果 (CN comment)
for j in range(2, 11):
    Xt, Yt = all_data[j-1]
    Xs = np.vstack([all_data[k-1][0] for k in range(1, j)])  # 合并前j-1个批次的数据
    Ys = np.vstack([all_data[k-1][1] for k in range(1, j)])

    n_components_options = [5, 10, 15, 20, 30, 40]
    alpha_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    temperature_options = [50, 100, 200]
    best_accuracy = 0
    best_n_components = None
    best_alpha = None
    best_temperature = None
    detailed_metrics = []

    for n_components in n_components_options:
        for alpha in alpha_options:
            for temperature in temperature_options:
                accuracies = []
                val_accuracies = []
                test_accuracies = []
                val_metrics = []
                test_metrics = []

                for experiment in range(1, 6):
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
                    loss, val_accuracy = student_model.evaluate(X_val_transformed, Y_val, verbose=0)
                    val_accuracies.append(val_accuracy)

                    Y_val_pred = student_model.predict(X_val_transformed)
                    Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
                    Y_val_classes = np.argmax(Y_val, axis=1)
                    val_f1 = f1_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                    val_precision = precision_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)
                    val_recall = recall_score(Y_val_classes, Y_val_pred_classes, average='macro', zero_division=1)

                    val_metrics.append({'Val Accuracy': val_accuracy, 'Val F1 Score': val_f1, 'Val Precision': val_precision, 'Val Recall': val_recall})

                    # Evaluate on the test set
                    loss, test_accuracy = student_model.evaluate(Xt_transformed, Y_test, verbose=0)
                    test_accuracies.append(test_accuracy)

                    Y_test_pred = student_model.predict(Xt_transformed)
                    Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
                    Y_test_classes = np.argmax(Y_test, axis=1)
                    test_f1 = f1_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                    test_precision = precision_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                    test_recall = recall_score(Y_test_classes, Y_test_pred_classes, average='macro', zero_division=1)
                    
                    test_metrics.append({'Test Accuracy': test_accuracy, 'Test F1 Score': test_f1, 'Test Precision': test_precision, 'Test Recall': test_recall})
                    
                    detailed_accuracies.append({'Model': f'student_model_{j}', 'Experiment': experiment, 'n_components': n_components, 'Alpha': alpha, 'Temperature': temperature, 'Val Accuracy': val_accuracy, 'Val F1 Score': val_f1, 'Val Precision': val_precision, 'Val Recall': val_recall, 'Test Accuracy': test_accuracy, 'Test F1 Score': test_f1, 'Test Precision': test_precision, 'Test Recall': test_recall})

                avg_val_accuracy = np.mean(val_accuracies)
                avg_val_f1 = np.mean([m['Val F1 Score'] for m in val_metrics])
                avg_val_precision = np.mean([m['Val Precision'] for m in val_metrics])
                avg_val_recall = np.mean([m['Val Recall'] for m in val_metrics])
                
                avg_test_accuracy = np.mean(test_accuracies)
                avg_test_f1 = np.mean([m['Test F1 Score'] for m in test_metrics])
                avg_test_precision = np.mean([m['Test Precision'] for m in test_metrics])
                avg_test_recall = np.mean([m['Test Recall'] for m in test_metrics])

                if avg_val_accuracy > best_accuracy:
                    best_accuracy = avg_val_accuracy
                    best_n_components = n_components
                    best_alpha = alpha
                    best_temperature = temperature
                    best_val_metrics = val_metrics
                    best_test_metrics = test_metrics

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
detailed_save_path = "outputs/DRCA_KD已有预测新的（定制版）详细3.xlsx"
detailed_results_df.to_excel(detailed_save_path, index=False)

# 保存汇总结果到Excel文件 (CN comment)
summary_save_path = "outputs/DRCA_KD已有预测新的（定制版）汇总3.xlsx"
summary_results_df.to_excel(summary_save_path, index=False)

# Print summary results进行检查 (CN comment)
print(summary_results_df)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Distillation loss function
def distillation_loss(y_true, y_pred, temperature):
    soft_true = tf.nn.softmax(y_true / temperature)
    soft_pred = tf.nn.softmax(y_pred / temperature)
    return tf.keras.losses.categorical_crossentropy(soft_true, soft_pred)

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
    summary_path = f"outputs/DRCA_KD已有预测新的（定制版）汇总_batch{j}.xlsx"
    summary_df = pd.read_excel(summary_path)
    
    best_n_components = summary_df['Best n_components'].values[0]
    best_alpha = summary_df['Best Alpha'].values[0]
    best_temperature = summary_df['Best Temperature'].values[0]
    
    detailed_accuracies = []
    Xt, Yt = all_data[j-1]
    Xs = np.vstack([all_data[k-1][0] for k in range(1, j)])  # 合并前j-1个批次的数据
    Ys = np.vstack([all_data[k-1][1] for k in range(1, j)])

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

#accuracy作图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
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
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch_num}.xlsx')
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
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_base_path = "outputs/"
drca_base_path = "outputs/"
kd_drca_base_path = "outputs/"

metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']

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
        kd_file_path = os.path.join(kd_base_path, f'知识蒸馏已有的预测新的详细_batch_{n}_30遍.xlsx')
        drca_file_path = os.path.join(drca_base_path, f'DRCA已有的预测新的详细_batch_{n}_30遍.xlsx')
        kd_drca_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD已有的预测新的详细_batch_{n}_30遍.xlsx')
        
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

#f1 score作图 (CN comment)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
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
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch_num}.xlsx')
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
plt.ylabel('F1 score', fontsize=32)

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
plt.ylabel('F1 score', fontsize=32)

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

#recall

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output file path
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
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
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch_num}.xlsx')
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
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
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
    batch_file_path = os.path.join(kd_drca_base_path, f'DRCA_KD已有预测新的（定制版）汇总_batch{batch_num}.xlsx')
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

# 绘制tSNE (CN comment)

# 绘制tSNE (CN comment)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 加载并处理数据 (CN comment)
def process_data(file_name):
    data = pd.read_csv(file_name, sep=' ', header=None)
    new_data = pd.DataFrame()
    for index, row in data.iterrows():
        gas_label = row[0]
        temp_dict = {"gas_label": gas_label}
        for item in row[1:]:
            if ':' in str(item):
                key, value = str(item).split(':')
                temp_dict[int(key)] = float(value)
        new_row = pd.DataFrame([temp_dict])
        new_data = pd.concat([new_data, new_row], ignore_index=True)
    new_data.reset_index(drop=True, inplace=True)
    return new_data

def prepare_data(data):
    Y = data['gas_label'].values - 1
    X = data.drop('gas_label', axis=1).values
    return X, Y

# Preprocess and cache all batches
all_data = []
for j in range(1, 11):
    data = process_data(f'data/raw/batch{j}.dat')
    X, Y = prepare_data(data)
    all_data.append((X, Y))

# Concatenate all batches to fit a global scaler
X_combined = np.vstack([data[0] for data in all_data])
Y_combined = np.hstack([data[1] for data in all_data])
batch_labels = np.hstack([[j] * len(data[1]) for j, data in enumerate(all_data, start=1)])

# Apply global standardization (fit on all data)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# 使用t-SNE降维 (CN comment)
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_combined_scaled)

# 创建DataFrame用于绘图 (CN comment)
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = Y_combined
tsne_df['Batch'] = batch_labels

# 设置字体大小 (CN comment)
plt.rcParams.update({'font.size': 24})

# 绘制整体数据的t-SNE图 (CN comment)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', palette='bright', data=tsne_df, legend='full', alpha=0.2, s=100)
plt.title('Overall t-SNE Visualization')
plt.xlabel('tSNE latent-1')
plt.ylabel('tSNE latent-2')
plt.legend(title='Class', loc='upper right', markerscale=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/overall_tsne_visualization.png', dpi=300)
plt.show()

# 绘制每个批次的t-SNE图 (CN comment)
for j in range(1, 11):
    plt.figure(figsize=(12, 8))
    batch_data = tsne_df[tsne_df['Batch'] == j]
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', palette='bright', data=batch_data, legend='full', alpha=0.2, s=100)
    plt.title(f'Batch {j}')
    plt.xlabel('tSNE latent varable-1')
    plt.ylabel('tSNE latent varable-2')
    plt.legend(title='Class', loc='upper right', markerscale=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/tsne_visualization_batch_{j}.png', dpi=300)
    plt.show()

#绘制显著性表格 (CN comment)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Output file path
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
kd_drca_summary_path = "outputs/DRCA_KD已有预测新的（定制版）汇总.xlsx"

# 指标列表 (CN comment)
metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']

# 读取数据 (CN comment)
df_baseline = pd.read_excel(baseline_path)
df_kd_summary = pd.read_excel(kd_summary_path)
df_drca_summary = pd.read_excel(drca_summary_path)
df_kd_drca_summary = pd.read_excel(kd_drca_summary_path)

# 处理数据函数 (CN comment)
def process_data(df, model_prefix, metric):
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)
    df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
    df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
    df = df.explode(metric)
    df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# 处理各个方法的数据 (CN comment)
def process_all_data(metric):
    baseline_data = process_data(df_baseline.copy(), 'model_', metric)
    kd_data = process_data(df_kd_summary.copy(), 'student_model_', metric)
    drca_data = process_data(df_drca_summary.copy(), 'drca_model_', metric)
    kd_drca_data = process_data(df_kd_drca_summary.copy(), 'student_model_', metric)
    return baseline_data, kd_data, drca_data, kd_drca_data

# Significance comparison helper
def compare_significance(baseline, method_data, metric):
    results = []
    for model in sorted(baseline['Model'].unique()):
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

# 创建最终结果表格 (CN comment)
def create_results_table():
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 设置表格数据框架 (CN comment)
    table_data = []
    for metric in metrics:
        baseline_data, kd_data, drca_data, kd_drca_data = process_all_data(metric)
        kd_results = compare_significance(baseline_data, kd_data, metric)
        drca_results = compare_significance(baseline_data, drca_data, metric)
        kd_drca_results = compare_significance(baseline_data, kd_drca_data, metric)
        
        table_data.append([metric.split()[1], 'KD'] + kd_results)
        table_data.append([metric.split()[1], 'DRCA'] + drca_results)
        table_data.append([metric.split()[1], 'KD_DRCA'] + kd_drca_results)

    # 绘制表格 (CN comment)
    columns = ['Metric', 'Method'] + [f'Test Batch {i}' for i in range(1, 10)]
    colors = {'Better': 'red', 'No Difference': 'blue', 'Worse': 'green', 'No Data': 'gray'}
    cell_colors = [[colors.get(val, 'white') for val in row] for row in table_data]

    table = ax.table(cellText=table_data, cellColours=cell_colors, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    ax.axis('off')

    # 图例 (CN comment)
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[key], label=key) for key in colors]
    plt.legend(handles=legend_patches, loc='lower right')

    plt.title('Significance Comparison of Different Methods for Various Metrics', fontsize=18)
    output_path = "outputs/"
    plt.savefig(output_path + "significance_comparison.png")
    plt.show()

# 调用函数 (CN comment)
create_results_table()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Output file path
baseline_path = "outputs/已有的预测新的详细.xlsx"
kd_summary_path = "outputs/知识蒸馏已有的预测新的（定制版）汇总.xlsx"
drca_summary_path = "outputs/DRCA已有的预测新的（定制版）汇总.xlsx"
kd_drca_summary_path = "outputs/DRCA_KD已有预测新的（定制版）汇总.xlsx"

# 指标列表 (CN comment)
metrics = ['Test Accuracy', 'Test F1 Score', 'Test Recall', 'Test Precision']

# 读取数据 (CN comment)
df_baseline = pd.read_excel(baseline_path)
df_kd_summary = pd.read_excel(kd_summary_path)
df_drca_summary = pd.read_excel(drca_summary_path)
df_kd_drca_summary = pd.read_excel(kd_drca_summary_path)

# 处理数据函数 (CN comment)
def process_data(df, model_prefix, metric):
    df['Model'] = df['Model'].astype(str).str.replace(model_prefix, '').astype(int)
    df[metric] = df[metric].astype(str)  # 将值转换为字符串格式
    df[metric] = df[metric].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
    df = df.explode(metric)
    df[metric] = df[metric].astype(float)  # 确保数据类型是浮点数
    return df

# 处理各个方法的数据 (CN comment)
def process_all_data(metric):
    baseline_data = process_data(df_baseline.copy(), 'model_', metric)
    kd_data = process_data(df_kd_summary.copy(), 'student_model_', metric)
    drca_data = process_data(df_drca_summary.copy(), 'drca_model_', metric)
    kd_drca_data = process_data(df_kd_drca_summary.copy(), 'student_model_', metric)
    return baseline_data, kd_data, drca_data, kd_drca_data

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

# 创建最终结果表格 (CN comment)
def create_results_table():
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 设置表格数据框架 (CN comment)
    table_data = []
    for method in ['KD', 'DRCA', 'KD_DRCA']:
        for metric in metrics:
            baseline_data, kd_data, drca_data, kd_drca_data = process_all_data(metric)
            if method == 'KD':
                method_data = kd_data
            elif method == 'DRCA':
                method_data = drca_data
            else:
                method_data = kd_drca_data
                
            results = compare_significance(baseline_data, method_data, metric)
            table_data.append([method, metric.split()[1]] + results)
    
    # 绘制表格 (CN comment)
    columns = ['Method', 'Metric'] + [f'Test Batch {i}' for i in range(2, 11)]
    colors = {'Better': 'red', 'No Difference': 'blue', 'Worse': 'green', 'No Data': 'gray'}
    cell_colors = [[colors.get(val, 'white') for val in row] for row in table_data]

    table = ax.table(cellText=table_data, cellColours=cell_colors, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1.5, 1.5)

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
import matplotlib.pyplot as plt

# Output file path
file_path1 = 'outputs/知识蒸馏一对一汇总test.xlsx'
file_path2 = 'outputs/知识蒸馏一对一汇总.xlsx'

# 读取Excel文件 (CN comment)
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

# 查看文件的列名 (CN comment)
print("File 1 Columns:", df1.columns)
print("File 2 Columns:", df2.columns)

# 假设列名可能是不同的 (CN comment)
model_column1 = 'Model'  # 假设列名为Model
accuracy_column1 = 'Avg Test Accuracy'  # 假设列名为Avg Test Accuracy

model_column2 = 'Model'  # 假设列名为Model
accuracy_column2 = 'Avg Test Accuracy'  # 假设列名为Avg Test Accuracy

# 获取数据 (CN comment)
models1 = df1[model_column1]
accuracy_values1 = df1[accuracy_column1]

models2 = df2[model_column2]
accuracy_values2 = df2[accuracy_column2]

# 绘制比较图 (CN comment)
plt.figure(figsize=(12, 8))

plt.plot(models1, accuracy_values1, marker='o', linestyle='-', label='File 1')
plt.plot(models2, accuracy_values2, marker='o', linestyle='-', label='File 2')

plt.xlabel('Model')
plt.ylabel('Avg Test Accuracy')
plt.title('Comparison of Avg Test Accuracy for Different Models')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
