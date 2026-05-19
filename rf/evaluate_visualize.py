"""
CICIDS2017 随机森林入侵检测模型 - 评估与可视化
==============================================
生成模型性能的可视化图表，包括：
1. 混淆矩阵热力图
2. 各类别精确率/召回率/F1对比图
3. Top特征重要性条形图
4. 类别分布饼图
5. ROC曲线 (一对多)
6. 分类性能雷达图
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
MODEL_DIR = r'd:\bishe\wgan\model'
OUTPUT_DIR = r'd:\bishe\wgan\evaluation_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (14, 10)

# 攻击类型中文映射
ATTACK_CN_NAMES = {
    'BENIGN': '正常流量',
    'Bot': '僵尸网络',
    'DDoS': 'DDoS攻击',
    'DoS GoldenEye': 'DoS GoldenEye',
    'DoS Hulk': 'DoS Hulk',
    'DoS Slowhttptest': 'DoS Slowhttptest',
    'DoS slowloris': 'DoS slowloris',
    'FTP-Patator': 'FTP暴力破解',
    'Heartbleed': 'Heartbleed漏洞',
    'Infiltration': '渗透攻击',
    'PortScan': '端口扫描',
    'SSH-Patator': 'SSH暴力破解',
    'Web Attack  Brute Force': 'Web暴力破解',
    'Web Attack  Sql Injection': 'SQL注入',
    'Web Attack  XSS': 'XSS攻击'
}

# 颜色方案
ATTACK_COLORS = {
    'BENIGN': '#2ECC71',
    'Bot': '#E74C3C',
    'DDoS': '#E67E22',
    'DoS GoldenEye': '#F39C12',
    'DoS Hulk': '#D35400',
    'DoS Slowhttptest': '#C0392B',
    'DoS slowloris': '#9B59B6',
    'FTP-Patator': '#3498DB',
    'Heartbleed': '#1ABC9C',
    'Infiltration': '#E91E63',
    'PortScan': '#00BCD4',
    'SSH-Patator': '#3F51B5',
    'Web Attack  Brute Force': '#FF5722',
    'Web Attack  Sql Injection': '#795548',
    'Web Attack  XSS': '#607D8B'
}


def load_data_and_model():
    """加载模型和测试数据"""
    print("=" * 60)
    print("加载模型和测试数据...")
    print("=" * 60)
    
    # 加载模型
    model_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    
    try:
        import joblib
        model = joblib.load(model_path)
    except:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # 加载预处理组件
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    
    # 加载特征重要性
    feature_importance = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'))
    
    print(f"模型加载完成! 类别数: {len(label_encoder.classes_)}")
    print(f"特征数: {len(feature_columns)}")
    
    return model, label_encoder, scaler, feature_columns, feature_importance


def load_test_data(scaler, feature_columns, label_encoder):
    """从原始数据集中加载测试集（使用分层采样）"""
    print("\n加载测试数据...")
    
    DATA_DIR = r'd:\bishe\traffic\CICIDS2017\MachineLearningCSV\MachineLearningCVE'
    
    # 加载所有数据
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    data_frames = []
    for f in all_files:
        filepath = os.path.join(DATA_DIR, f)
        df = pd.read_csv(filepath)
        data_frames.append(df)
    
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # 数据清洗
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.dropna(inplace=True)
    
    # 分离特征和标签
    X = df_all.drop(columns=[' Label'])
    y = df_all[' Label']
    
    # 特征处理
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
    
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    
    # 只保留模型训练时使用的特征列
    X = X[feature_columns]
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 标签编码
    y_encoded = label_encoder.transform(y)
    
    # 分层采样取20%作为测试集
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"测试集类别分布:")
    for cls_id in sorted(np.unique(y_test)):
        cls_name = label_encoder.classes_[cls_id]
        count = np.sum(y_test == cls_id)
        print(f"  {cls_name:30s}: {count}")
    
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵热力图"""
    print("\n生成混淆矩阵热力图...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # 原始数值混淆矩阵
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    ax1.set_title('混淆矩阵 (原始计数)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # 归一化混淆矩阵（百分比）
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8}, vmin=0, vmax=1)
    ax2.set_title('混淆矩阵 (归一化百分比)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_ylabel('真实标签', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_classification_metrics(y_true, y_pred, class_names, save_path):
    """绘制各类别的精确率、召回率、F1分数对比图"""
    print("生成分类指标对比图...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for cls in class_names:
        if cls in report:
            classes.append(cls)
            precisions.append(report[cls]['precision'])
            recalls.append(report[cls]['recall'])
            f1_scores.append(report[cls]['f1-score'])
            supports.append(report[cls]['support'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    bars1 = ax.bar(x - width, precisions, width, label='精确率 (Precision)', color='#3498DB')
    bars2 = ax.bar(x, recalls, width, label='召回率 (Recall)', color='#2ECC71')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1分数', color='#E74C3C')
    
    # 在柱状图上标注数值
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    add_labels(bars1, precisions)
    add_labels(bars2, recalls)
    add_labels(bars3, f1_scores)
    
    ax.set_xlabel('攻击类型', fontsize=14)
    ax.set_ylabel('分数', fontsize=14)
    ax.set_title('各类别精确率/召回率/F1分数对比', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加支持样本数标注
    for i, (cls, sup) in enumerate(zip(classes, supports)):
        ax.annotate(f'n={sup}', xy=(i, 0.02), ha='center', fontsize=7, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_feature_importance(feature_importance, save_path, top_n=20):
    """绘制Top N特征重要性条形图"""
    print(f"生成Top{top_n}特征重要性图...")
    
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, 
                   color=colors[::-1])
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('重要性得分', fontsize=14)
    ax.set_title(f'Top {top_n} 最重要的网络流量特征', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 在条形上标注数值
    for bar, val in zip(bars, top_features['importance'].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_class_distribution(y_true, class_names, save_path):
    """绘制测试集类别分布饼图"""
    print("生成类别分布图...")
    
    from collections import Counter
    counts = Counter(y_true)
    
    labels = []
    sizes = []
    colors = []
    explode = []
    
    for cls_id in sorted(counts.keys()):
        cls_name = class_names[cls_id]
        labels.append(cls_name)
        sizes.append(counts[cls_id])
        color = ATTACK_COLORS.get(cls_name, '#95A5A6')
        colors.append(color)
        explode.append(0.05 if cls_name != 'BENIGN' else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 饼图
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=None, autopct='%1.1f%%',
        startangle=90, colors=colors, explode=explode,
        pctdistance=0.85, textprops={'fontsize': 8}
    )
    ax1.set_title('测试集流量类别分布', fontsize=16, fontweight='bold')
    
    # 添加图例
    legend_labels = [f'{l} ({s:,})' for l, s in zip(labels, sizes)]
    ax1.legend(wedges, legend_labels, title="流量类型", loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    # 条形图（对数尺度）
    ax2.barh(range(len(labels)), sizes, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xlabel('样本数量 (对数尺度)', fontsize=12)
    ax2.set_title('各类别样本数量 (对数尺度)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 在条形上标注数值
    for i, (label, size) in enumerate(zip(labels, sizes)):
        ax2.text(size * 1.1, i, f'{size:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_roc_curves(model, X_test, y_test, class_names, save_path):
    """绘制一对多ROC曲线"""
    print("生成ROC曲线...")
    
    n_classes = len(class_names)
    
    # 将标签二值化
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # 获取预测概率
    y_score = model.predict_proba(X_test)
    
    # 计算每个类别的ROC曲线和AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 只对样本数较多的类别绘制ROC曲线（避免稀疏问题）
    min_samples_for_roc = 10
    
    for i in range(n_classes):
        class_sample_count = np.sum(y_test == i)
        if class_sample_count >= min_samples_for_roc:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            cls_name = class_names[i]
            color = ATTACK_COLORS.get(cls_name, '#95A5A6')
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{cls_name} (AUC={roc_auc[i]:.3f})')
    
    # 随机猜测基线
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=14)
    ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=14)
    ax.set_title('一对多 ROC 曲线', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_radar_chart(y_true, y_pred, class_names, save_path):
    """绘制分类性能雷达图"""
    print("生成性能雷达图...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 筛选出有足够样本的类别
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for cls in class_names:
        if cls in report and report[cls]['support'] >= 5:
            classes.append(cls)
            precisions.append(report[cls]['precision'])
            recalls.append(report[cls]['recall'])
            f1_scores.append(report[cls]['f1-score'])
    
    if len(classes) == 0:
        print("  跳过雷达图：没有足够的类别")
        return
    
    # 准备雷达图数据
    N = len(classes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    precisions_closed = precisions + precisions[:1]
    recalls_closed = recalls + recalls[:1]
    f1_closed = f1_scores + f1_scores[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    
    ax.plot(angles, precisions_closed, 'o-', linewidth=2, label='精确率', color='#3498DB')
    ax.fill(angles, precisions_closed, alpha=0.1, color='#3498DB')
    
    ax.plot(angles, recalls_closed, 'o-', linewidth=2, label='召回率', color='#2ECC71')
    ax.fill(angles, recalls_closed, alpha=0.1, color='#2ECC71')
    
    ax.plot(angles, f1_closed, 'o-', linewidth=2, label='F1分数', color='#E74C3C')
    ax.fill(angles, f1_closed, alpha=0.1, color='#E74C3C')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('各类别分类性能雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_error_analysis(y_true, y_pred, class_names, save_path):
    """绘制错误分析图 - 每个类别的错误分类情况"""
    print("生成错误分析图...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的错误数（真正例在混淆矩阵对角线之外）
    n_classes = len(class_names)
    errors_per_class = []
    
    for i in range(n_classes):
        total = cm[i, :].sum()
        correct = cm[i, i]
        errors = total - correct
        errors_per_class.append(errors)
    
    # 计算错误率
    totals = cm.sum(axis=1)
    error_rates = np.where(totals > 0, errors_per_class / totals, 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 错误数量条形图
    colors = ['#E74C3C' if e > 0 else '#2ECC71' for e in errors_per_class]
    bars = ax1.bar(range(n_classes), errors_per_class, color=colors)
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('错误分类数量', fontsize=12)
    ax1.set_title('各类别错误分类数量', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, errors_per_class):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(errors_per_class)*0.01,
                    f'{val}', ha='center', va='bottom', fontsize=9)
    
    # 错误率条形图
    colors2 = ['#E74C3C' if e > 0.05 else '#F39C12' if e > 0 else '#2ECC71' for e in error_rates]
    bars2 = ax2.bar(range(n_classes), error_rates * 100, color=colors2)
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('错误率 (%)', fontsize=12)
    ax2.set_title('各类别错误率', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, error_rates):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_performance_summary(y_true, y_pred, class_names, save_path):
    """绘制性能摘要仪表盘"""
    print("生成性能摘要仪表盘...")
    
    # 计算总体指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 宏平均
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_data = [
        ('加权平均', [('精确率', prec), ('召回率', rec), ('F1分数', f1), ('准确率', acc)],
         ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']),
        ('宏平均', [('精确率', prec_macro), ('召回率', rec_macro), ('F1分数', f1_macro)],
         ['#3498DB', '#2ECC71', '#E74C3C'])
    ]
    
    for idx, (title, metrics, colors) in enumerate(metrics_data):
        ax = axes[0, idx]
        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 准确率仪表盘（大数字显示）
    ax = axes[0, 2]
    ax.axis('off')
    ax.text(0.5, 0.7, f'{acc*100:.2f}%', ha='center', va='center',
            fontsize=48, fontweight='bold', color='#2ECC71')
    ax.text(0.5, 0.3, '总体准确率', ha='center', va='center',
            fontsize=18, fontweight='bold')
    
    # 底部：各类别F1分数热力图
    ax = axes[1, :]
    ax = plt.subplot(2, 3, (4, 6))
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    classes_filtered = []
    f1_values = []
    support_values = []
    
    for cls in class_names:
        if cls in report:
            classes_filtered.append(cls)
            f1_values.append(report[cls]['f1-score'])
            support_values.append(report[cls]['support'])
    
    # 按F1分数排序
    sorted_indices = np.argsort(f1_values)
    classes_sorted = [classes_filtered[i] for i in sorted_indices]
    f1_sorted = [f1_values[i] for i in sorted_indices]
    support_sorted = [support_values[i] for i in sorted_indices]
    
    colors_f1 = ['#E74C3C' if f < 0.8 else '#F39C12' if f < 0.95 else '#2ECC71' for f in f1_sorted]
    
    bars = ax.barh(range(len(classes_sorted)), f1_sorted, color=colors_f1)
    ax.set_yticks(range(len(classes_sorted)))
    ax.set_yticklabels(classes_sorted, fontsize=9)
    ax.set_xlabel('F1分数', fontsize=12)
    ax.set_title('各类别F1分数 (按升序排列)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, f1_val, sup in zip(bars, f1_sorted, support_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
               f'{f1_val:.3f} (n={sup})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def main():
    print("\n" + "=" * 60)
    print("CICIDS2017 随机森林模型评估与可视化")
    print("=" * 60)
    
    # 加载模型和数据
    model, label_encoder, scaler, feature_columns, feature_importance = load_data_and_model()
    X_test, y_test = load_test_data(scaler, feature_columns, label_encoder)
    
    # 预测
    print("\n进行预测...")
    y_pred = model.predict(X_test)
    
    class_names = label_encoder.classes_
    
    # 生成所有可视化图表
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    # 1. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, class_names,
                         os.path.join(OUTPUT_DIR, '1_confusion_matrix.png'))
    
    # 2. 分类指标对比
    plot_classification_metrics(y_test, y_pred, class_names,
                               os.path.join(OUTPUT_DIR, '2_classification_metrics.png'))
    
    # 3. 特征重要性
    plot_feature_importance(feature_importance,
                           os.path.join(OUTPUT_DIR, '3_feature_importance.png'))
    
    # 4. 类别分布
    plot_class_distribution(y_test, class_names,
                           os.path.join(OUTPUT_DIR, '4_class_distribution.png'))
    
    # 5. ROC曲线
    plot_roc_curves(model, X_test, y_test, class_names,
                   os.path.join(OUTPUT_DIR, '5_roc_curves.png'))
    
    # 6. 雷达图
    plot_radar_chart(y_test, y_pred, class_names,
                    os.path.join(OUTPUT_DIR, '6_radar_chart.png'))
    
    # 7. 错误分析
    plot_error_analysis(y_test, y_pred, class_names,
                       os.path.join(OUTPUT_DIR, '7_error_analysis.png'))
    
    # 8. 性能摘要
    plot_performance_summary(y_test, y_pred, class_names,
                            os.path.join(OUTPUT_DIR, '8_performance_summary.png'))
    
    print("\n" + "=" * 60)
    print(f"所有可视化图表已生成到: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n生成的文件列表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f:45s} ({size/1024:.1f} KB)")


if __name__ == '__main__':
    main()
