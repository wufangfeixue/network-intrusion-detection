# CICIDS2017 Network Intrusion Detection

基于CICIDS2017数据集的随机森林网络入侵检测系统。

## 项目概述

本项目使用随机森林算法对CICIDS2017网络流量数据进行训练，实现对正常流量和多种网络攻击类型的自动识别与分类。

## 数据集说明

CICIDS2017数据集包含以下攻击类型：

| 攻击类型 | 英文名称 | 说明 |
|---------|---------|------|
| 正常流量 | BENIGN | 正常网络流量 |
| 僵尸网络 | Bot | Botnet攻击 |
| 分布式拒绝服务 | DDoS | 分布式拒绝服务攻击 |
| DoS GoldenEye | DoS GoldenEye | DoS GoldenEye攻击 |
| DoS Hulk | DoS Hulk | DoS Hulk攻击 |
| DoS Slowhttptest | DoS Slowhttptest | DoS Slowhttptest攻击 |
| DoS slowloris | DoS slowloris | DoS slowloris攻击 |
| FTP暴力破解 | FTP-Patator | FTP密码暴力破解攻击 |
| Heartbleed | Heartbleed | Heartbleed漏洞攻击 |
| 渗透攻击 | Infiltration | 网络渗透攻击 |
| 端口扫描 | PortScan | 端口扫描攻击 |
| SSH暴力破解 | SSH-Patator | SSH密码暴力破解攻击 |
| Web暴力破解 | Web Attack Brute Force | Web暴力破解攻击 |
| SQL注入 | Web Attack Sql Injection | SQL注入攻击 |
| XSS跨站脚本 | Web Attack XSS | XSS跨站脚本攻击 |

## 项目结构

```
.
├── train_rf_model.py       # 模型训练脚本
├── evaluate_visualize.py   # 模型评估与可视化脚本
├── predict.py              # 预测/推理脚本
├── dataset_info.txt        # 数据集信息
├── model/                  # 训练好的模型文件
│   ├── rf_model.pkl        # 随机森林模型
│   ├── rf_model.joblib     # joblib格式模型
│   ├── label_encoder.pkl   # 标签编码器
│   ├── scaler.pkl          # 特征标准化器
│   ├── feature_columns.pkl  # 特征列名
│   └── feature_importance.csv # 特征重要性
└── evaluation_output/       # 评估可视化结果
    ├── confusion_matrix.png
    ├── classification_metrics.png
    ├── feature_importance.png
    ├── class_distribution.png
    ├── roc_curves.png
    ├── radar_chart.png
    ├── error_analysis.png
    └── performance_summary.png
```

## 环境要求

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train_rf_model.py
```

训练完成后，模型文件将保存在 `model/` 目录下。

### 2. 评估模型

```bash
python evaluate_visualize.py
```

评估结果和可视化图表将保存在 `evaluation_output/` 目录下。

### 3. 预测新数据

命令行预测：
```bash
python predict.py --input test_data.csv --output results.csv
```

交互式预测：
```bash
python predict.py --interactive
```

## 模型性能

模型在测试集上的表现：

- **准确率 (Accuracy)**: >99%
- **精确率 (Precision)**: >99%
- **召回率 (Recall)**: >99%
- **F1分数**: >99%

## 攻击检测结果

| 攻击类型 | 检测能力 |
|---------|---------|
| DDoS | ✅ 高检测率 |
| PortScan | ✅ 高检测率 |
| Bot | ✅ 高检测率 |
| DoS攻击 | ✅ 高检测率 |
| Brute Force | ✅ 高检测率 |
| Web攻击 | ✅ 高检测率 |
| Heartbleed | ✅ 可检测 |
| Infiltration | ⚠️ 样本较少 |

## 特征工程

模型使用78个网络流量特征，包括：

- 流量统计特征
- 连接特征
- 内容特征
- 时间特征

最重要的特征包括：
1. Fwd Packet Length Mean
2. Packet Length Mean
3. Packet Length Std
4. Flow IAT Mean
5. etc.

## 注意事项

1. 数据集路径默认为 `d:\bishe\traffic\CICIDS2017\MachineLearningCSV\MachineLearningCVE`
2. 模型输出路径默认为 `d:\bishe\wgan\model`
3. 如需更换路径，请修改脚本中的 `DATA_DIR` 和 `MODEL_DIR` 变量

## 许可证

MIT License

## 参考

- CICIDS2017 Dataset: https://www.unb.ca/cic/datasets/ids2017.html
