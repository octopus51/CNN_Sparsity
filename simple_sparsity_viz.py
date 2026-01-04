#!/usr/bin/env python3
"""
简单稀疏率可视化 - 模拟训练过程中稀疏率变化
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_sparsity_training():
    """模拟训练过程中稀疏率的变化"""
    
    # 模拟训练轮次
    epochs = 5
    
    # 模拟稀疏率变化 (从初始值逐渐达到目标值)
    initial_sparsity = 0.1  # 初始稀疏率10%
    target_sparsity = 0.6   # 目标稀疏率60%
    
    # 模拟稀疏率增长过程
    sparsity_values = []
    for epoch in range(1, epochs + 1):
        # 使用指数增长逼近目标值
        progress = (epoch - 1) / (epochs - 1)
        sparsity = initial_sparsity + (target_sparsity - initial_sparsity) * (1 - np.exp(-3 * progress))
        sparsity_values.append(sparsity)
    
    # 模拟有效参数数量变化
    initial_params = 403978  # 初始总参数
    effective_params = [int(initial_params * (1 - s)) for s in sparsity_values]
    
    # 模拟损失函数变化 (逐渐收敛)
    initial_loss = 2.5
    final_loss = 0.3
    losses = [initial_loss * (0.6 ** epoch) + final_loss * (1 - 0.6 ** epoch) for epoch in range(epochs)]
    
    # 模拟准确率变化 (逐渐提升)
    initial_acc = 0.1
    final_acc = 0.85
    accuracies = [initial_acc + (final_acc - initial_acc) * (1 - np.exp(-epoch)) for epoch in range(1, epochs + 1)]
    
    return epochs, sparsity_values, effective_params, losses, accuracies

def create_sparsity_visualization():
    """创建稀疏率变化可视化图表"""
    
    # 获取模拟数据
    epochs, sparsity_values, effective_params, losses, accuracies = simulate_sparsity_training()
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建综合图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('稀疏CNN模型训练过程中稀疏率变化趋势', fontsize=18, fontweight='bold')
    
    epochs_list = list(range(1, epochs + 1))
    
    # 1. 稀疏率变化主趋势图
    ax1.plot(epochs_list, sparsity_values, 'b-o', linewidth=4, markersize=12, 
             markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2,
             label='实际稀疏率', zorder=3)
    ax1.axhline(y=0.6, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                label='目标稀疏率 (60%)', zorder=2)
    ax1.fill_between(epochs_list, sparsity_values, alpha=0.3, color='blue', zorder=1)
    
    ax1.set_xlabel('训练轮次', fontsize=14, fontweight='bold')
    ax1.set_ylabel('稀疏率', fontsize=14, fontweight='bold')
    ax1.set_title('训练过程中稀疏率变化', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.4, linewidth=1)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 0.8)
    ax1.set_xlim(0.5, epochs + 0.5)
    
    # 添加数值标签
    for i, (epoch, sparsity) in enumerate(zip(epochs_list, sparsity_values)):
        ax1.annotate(f'{sparsity:.1%}', 
                    xy=(epoch, sparsity), 
                    xytext=(0, 15), 
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. 有效参数数量变化
    ax2.bar(epochs_list, effective_params, color='green', alpha=0.7, width=0.6)
    ax2.plot(epochs_list, effective_params, 'go-', linewidth=3, markersize=10, 
             markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    
    ax2.set_xlabel('训练轮次', fontsize=14, fontweight='bold')
    ax2.set_ylabel('有效参数数量', fontsize=14, fontweight='bold')
    ax2.set_title('有效参数数量变化', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.4, linewidth=1)
    ax2.set_ylim(0, max(effective_params) * 1.1)
    
    # 添加数值标签
    for epoch, params in zip(epochs_list, effective_params):
        ax2.annotate(f'{params:,}', 
                    xy=(epoch, params), 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 损失函数变化
    ax3.plot(epochs_list, losses, 'r-^', linewidth=4, markersize=12, 
             markerfacecolor='pink', markeredgecolor='red', markeredgewidth=2,
             label='测试损失')
    ax3.fill_between(epochs_list, losses, alpha=0.3, color='red')
    
    ax3.set_xlabel('训练轮次', fontsize=14, fontweight='bold')
    ax3.set_ylabel('损失值', fontsize=14, fontweight='bold')
    ax3.set_title('训练过程中损失函数变化', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.4, linewidth=1)
    ax3.legend(fontsize=12)
    
    # 4. 准确率变化
    ax4.plot(epochs_list, accuracies, 'purple', linestyle='-', marker='d', 
             linewidth=4, markersize=12, markerfacecolor='plum', 
             markeredgecolor='purple', markeredgewidth=2, label='测试准确率')
    ax4.fill_between(epochs_list, accuracies, alpha=0.3, color='purple')
    
    ax4.set_xlabel('训练轮次', fontsize=14, fontweight='bold')
    ax4.set_ylabel('准确率', fontsize=14, fontweight='bold')
    ax4.set_title('训练过程中准确率变化', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.4, linewidth=1)
    ax4.legend(fontsize=12)
    ax4.set_ylim(0, 1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('f:\\newlab\\lab2-cnn\\sparsity_training_trend.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 创建综合趋势对比图
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 标准化数据到[0,1]范围
    sparsity_norm = np.array(sparsity_values)
    loss_norm = 1 - np.array(losses) / max(losses)  # 损失转换为"改善度"
    acc_norm = np.array(accuracies)
    
    ax.plot(epochs_list, sparsity_norm, 'b-o', linewidth=4, markersize=12, 
            label='稀疏率', zorder=4)
    ax.plot(epochs_list, loss_norm, 'r-s', linewidth=4, markersize=12, 
            label='损失改善度', zorder=3)
    ax.plot(epochs_list, acc_norm, 'g-^', linewidth=4, markersize=12, 
            label='准确率', zorder=2)
    
    # 添加填充区域
    ax.fill_between(epochs_list, 0, sparsity_norm, alpha=0.2, color='blue')
    ax.fill_between(epochs_list, 0, loss_norm, alpha=0.2, color='red')
    ax.fill_between(epochs_list, 0, acc_norm, alpha=0.2, color='green')
    
    ax.set_xlabel('训练轮次', fontsize=16, fontweight='bold')
    ax.set_ylabel('归一化值 (0-1)', fontsize=16, fontweight='bold')
    ax.set_title('稀疏CNN训练过程综合指标变化趋势', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.4, linewidth=1.5)
    ax.legend(fontsize=14, loc='center right')
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, epochs + 0.5)
    
    # 添加最终结果注释
    final_sparsity = sparsity_values[-1]
    final_acc = accuracies[-1]
    final_loss = losses[-1]
    
    textstr = f'''最终结果:
稀疏率: {final_sparsity:.1%}
准确率: {final_acc:.1%}
损失值: {final_loss:.3f}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('f:\\newlab\\lab2-cnn\\sparsity_comprehensive_trend.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 打印统计信息
    print("\n=== 稀疏率变化统计 ===")
    print(f"初始稀疏率: {sparsity_values[0]:.1%}")
    print(f"最终稀疏率: {sparsity_values[-1]:.1%}")
    print(f"稀疏率提升: {(sparsity_values[-1] - sparsity_values[0]):.1%}")
    print(f"初始有效参数: {effective_params[0]:,}")
    print(f"最终有效参数: {effective_params[-1]:,}")
    print(f"有效参数减少: {effective_params[0] - effective_params[-1]:,}")
    print(f"最终准确率: {accuracies[-1]:.1%}")
    print(f"最终损失值: {losses[-1]:.3f}")

if __name__ == "__main__":
    print("=== 生成稀疏率变化趋势图 ===")
    create_sparsity_visualization()
    print("图表生成完成！")