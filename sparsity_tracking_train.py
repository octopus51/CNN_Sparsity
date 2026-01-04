#!/usr/bin/env python3
"""
稀疏率跟踪训练脚本 - 记录训练过程中稀疏率的变化
"""

from cnn_mnist_sparse import SparseCNN, load_and_preprocess_mnist
import numpy as np
import matplotlib.pyplot as plt
import time

class SparseTracker:
    def __init__(self):
        self.epoch_sparsity = []
        self.epoch_effective_params = []
        self.epoch_total_params = []
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.epoch_times = []
    
    def record_epoch(self, epoch, model, test_loss, test_acc, epoch_time):
        """记录每个epoch的稀疏率信息"""
        current_sparsity = model._get_current_sparsity()
        
        # 计算总参数和有效参数
        total_params = 0
        effective_params = 0
        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                total_params += layer.weights.size
                effective_params += np.sum(np.abs(layer.weights) > 0)
            if hasattr(layer, 'bias') and layer.bias is not None:
                total_params += layer.bias.size
                effective_params += np.sum(np.abs(layer.bias) > 0)
        
        self.epoch_sparsity.append(current_sparsity)
        self.epoch_effective_params.append(effective_params)
        self.epoch_total_params.append(total_params)
        self.epoch_losses.append(test_loss)
        self.epoch_accuracies.append(test_acc)
        self.epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch}: 稀疏率={current_sparsity:.4f}, 有效参数={effective_params}, 总参数={total_params}")

def main():
    print("=== 稀疏率跟踪训练 ===")
    
    # 加载数据（使用十分之一数据）
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist(use_small_data=True, small_data_size=6000)
    print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")
    
    # 创建稀疏CNN模型
    print("\n创建稀疏CNN模型...")
    sparse_model = SparseCNN(input_shape=(1, 28, 28, 1), num_classes=10)
    
    # 初始化跟踪器
    tracker = SparseTracker()
    
    # 训练参数
    epochs = 3
    batch_size = 32
    learning_rate = 0.01
    
    print(f"\n开始训练，共 {epochs} 轮...")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_start = time.time()
        
        # 执行一轮训练
        results = sparse_model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=1,  # 只训练1轮
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        epoch_time = time.time() - epoch_start
        
        # 记录结果
        test_loss = results['test_losses'][-1]
        test_acc = results['test_accuracies'][-1]
        tracker.record_epoch(epoch+1, sparse_model, test_loss, test_acc, epoch_time)
    
    # 生成可视化图表
    print("\n生成训练过程可视化...")
    create_sparsity_visualization(tracker, epochs)
    
    print("训练完成！稀疏率变化图表已保存。")

def create_sparsity_visualization(tracker, epochs):
    """创建稀疏率变化的可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('稀疏CNN模型训练过程分析', fontsize=16, fontweight='bold')
    
    epochs_list = list(range(1, epochs + 1))
    
    # 1. 稀疏率变化趋势
    ax1.plot(epochs_list, tracker.epoch_sparsity, 'b-o', linewidth=2, markersize=8, label='稀疏率')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('稀疏率')
    ax1.set_title('训练过程中稀疏率变化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 添加目标线
    ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='目标稀疏率(60%)')
    ax1.legend()
    
    # 2. 参数数量变化
    ax2.plot(epochs_list, tracker.epoch_total_params, 'g-s', linewidth=2, markersize=8, label='总参数')
    ax2.plot(epochs_list, tracker.epoch_effective_params, 'orange', linestyle='-', marker='o', linewidth=2, markersize=8, label='有效参数')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('参数数量')
    ax2.set_title('训练过程中参数数量变化')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 损失函数变化
    ax3.plot(epochs_list, tracker.epoch_losses, 'r-^', linewidth=2, markersize=8, label='测试损失')
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('损失值')
    ax3.set_title('训练过程中损失函数变化')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 准确率变化
    ax4.plot(epochs_list, tracker.epoch_accuracies, 'purple', linestyle='-', marker='d', linewidth=2, markersize=8, label='测试准确率')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('准确率')
    ax4.set_title('训练过程中准确率变化')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('f:\\newlab\\lab2-cnn\\sparsity_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建综合趋势图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 标准化数据进行对比
    sparsity_norm = np.array(tracker.epoch_sparsity)
    loss_norm = 1 - np.array(tracker.epoch_losses) / max(tracker.epoch_losses)  # 归一化损失
    acc_norm = np.array(tracker.epoch_accuracies)
    effective_ratio = np.array(tracker.epoch_effective_params) / np.array(tracker.epoch_total_params)
    
    ax.plot(epochs_list, sparsity_norm, 'b-o', linewidth=3, markersize=10, label='稀疏率')
    ax.plot(epochs_list, loss_norm, 'r-s', linewidth=3, markersize=10, label='损失改善度')
    ax.plot(epochs_list, acc_norm, 'g-^', linewidth=3, markersize=10, label='准确率')
    ax.plot(epochs_list, effective_ratio, 'orange', linestyle='-', marker='d', linewidth=3, markersize=10, label='有效参数比例')
    
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('归一化值', fontsize=12)
    ax.set_title('稀疏CNN训练过程综合指标变化', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    
    # 添加注释
    final_sparsity = tracker.epoch_sparsity[-1]
    final_acc = tracker.epoch_accuracies[-1]
    ax.annotate(f'最终稀疏率: {final_sparsity:.2%}', 
                xy=(epochs, final_sparsity), 
                xytext=(epochs-0.3, final_sparsity+0.1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=12, color='blue')
    
    plt.tight_layout()
    plt.savefig('f:\\newlab\\lab2-cnn\\sparsity_comprehensive_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print("\n=== 训练过程统计 ===")
    print(f"初始稀疏率: {tracker.epoch_sparsity[0]:.4f}")
    print(f"最终稀疏率: {tracker.epoch_sparsity[-1]:.4f}")
    print(f"稀疏率提升: {(tracker.epoch_sparsity[-1] - tracker.epoch_sparsity[0]):.4f}")
    print(f"初始有效参数: {tracker.epoch_effective_params[0]:,}")
    print(f"最终有效参数: {tracker.epoch_effective_params[-1]:,}")
    print(f"最终准确率: {tracker.epoch_accuracies[-1]:.4f}")
    print(f"总训练时间: {sum(tracker.epoch_times):.2f}秒")

if __name__ == "__main__":
    main()