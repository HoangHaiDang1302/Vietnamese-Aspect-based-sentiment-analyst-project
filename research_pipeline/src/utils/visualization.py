"""
Visualization Module
Vẽ biểu đồ cho báo cáo nghiên cứu: Training curves, Comparison bars, Confusion matrix, Error analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style='whitegrid')


def plot_training_curves(histories, save_path=None):
    """Vẽ 3 biểu đồ: Train Loss, Dev Loss, Dev Metric theo Epoch cho nhiều models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train Loss
    for name, hist in histories.items():
        axes[0].plot(hist['train_loss'], label=name, marker='o', markersize=3)
    axes[0].set_title('Training Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)

    # Dev Loss
    for name, hist in histories.items():
        axes[1].plot(hist['dev_loss'], label=name, marker='s', markersize=3)
    axes[1].set_title('Validation Loss', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=8)

    # Dev Metric (tok_acc hoặc dev_acc)
    for name, hist in histories.items():
        metric_key = 'dev_tok_acc' if 'dev_tok_acc' in hist else 'dev_acc'
        if metric_key in hist:
            axes[2].plot(hist[metric_key], label=name, marker='^', markersize=3)
    axes[2].set_title('Validation Metric', fontsize=13)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_df, metric_col='F1', title='Model Comparison', save_path=None):
    """Vẽ barplot so sánh F1 giữa các models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(results_df))
    bars = ax.bar(results_df['Model'], results_df[metric_col], color=colors)

    # Ghi số liệu trên đầu cột
    for bar, val in zip(bars, results_df[metric_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_ylim(0, min(1.0, results_df[metric_col].max() + 0.1))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """Vẽ Confusion Matrix cho ASC."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_label_f1(metrics_dict, label_names, title='Per-Label F1 Score', save_path=None):
    """Vẽ horizontal barplot cho F1 score từng nhãn (dùng cho E2E hoặc ATE)."""
    f1_scores = [metrics_dict[ln]['f1'] for ln in label_names if ln in metrics_dict]
    supports = [metrics_dict[ln]['support'] for ln in label_names if ln in metrics_dict]
    valid_labels = [ln for ln in label_names if ln in metrics_dict]

    fig, ax = plt.subplots(figsize=(10, max(6, len(valid_labels) * 0.35)))
    colors = ['#e74c3c' if s < 20 else '#3498db' for s in supports]
    bars = ax.barh(valid_labels, f1_scores, color=colors)

    for bar, f1, sup in zip(bars, f1_scores, supports):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{f1:.3f} (n={sup})', va='center', fontsize=8)

    ax.set_xlabel('F1 Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_analysis(error_df, top_n=10, save_path=None):
    """Vẽ biểu đồ phân tích lỗi: Nhãn nào bị nhầm nhiều nhất."""
    if 'Error_Type' in error_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        error_counts = error_df['Error_Type'].value_counts().head(top_n)
        sns.barplot(x=error_counts.values, y=error_counts.index, palette='Reds_r', ax=ax)
        ax.set_title(f'Top {top_n} Error Types', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
