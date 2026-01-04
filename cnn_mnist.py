import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request

# --------------------------- 激活函数 ---------------------------
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Softmax:
    def forward(self, x):
        # 数值稳定的softmax实现
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def backward(self, grad_output):
        # 通常与交叉熵损失结合使用，直接从损失函数获取梯度
        return grad_output

# --------------------------- 损失函数 ---------------------------
class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        # 数值稳定的交叉熵计算
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    
    def backward(self):
        # 计算梯度
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]

# --------------------------- 卷积层 ---------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重和偏置
        variance = 2.0 / (in_channels * kernel_size * kernel_size)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(variance)
        self.bias = np.zeros(out_channels)
        
        # 存储梯度
        self.grad_weights = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        batch_size, in_height, in_width, in_channels = x.shape
        
        # 计算输出形状
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        padded_x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # 初始化输出
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        # 执行卷积
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c_out in range(self.out_channels):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取感受野
                        receptive_field = padded_x[b, h_start:h_end, w_start:w_end, :]
                        
                        # 卷积操作（调整权重形状以匹配感受野）
                        weight_reshaped = np.transpose(self.weights[c_out], (1, 2, 0))  # 从 (in_channels, kernel_h, kernel_w) 转置为 (kernel_h, kernel_w, in_channels)
                        output[b, h, w, c_out] = np.sum(receptive_field * weight_reshaped) + self.bias[c_out]
        
        return output
    
    def backward(self, grad_output):
        batch_size, out_height, out_width, out_channels = grad_output.shape
        batch_size, in_height, in_width, in_channels = self.input.shape
        
        # 初始化梯度
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.input)
        
        # 填充输入和梯度输入
        padded_x = np.pad(self.input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        padded_grad_input = np.pad(grad_input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # 计算梯度
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = w * self.stride
                    w_end = w_start + self.kernel_size
                    
                    for c_out in range(out_channels):
                        # 计算权重梯度（调整输入形状以匹配权重）
                        receptive_field = padded_x[b, h_start:h_end, w_start:w_end, :]
                        receptive_field_reshaped = np.transpose(receptive_field, (2, 0, 1))  # 从 (kernel_h, kernel_w, in_channels) 转置为 (in_channels, kernel_h, kernel_w)
                        self.grad_weights[c_out] += receptive_field_reshaped * grad_output[b, h, w, c_out]
                        
                        # 计算偏置梯度
                        self.grad_bias[c_out] += grad_output[b, h, w, c_out]
                        
                        # 计算输入梯度（调整权重形状以匹配输入）
                        weight_reshaped = np.transpose(self.weights[c_out], (1, 2, 0))  # 从 (in_channels, kernel_h, kernel_w) 转置为 (kernel_h, kernel_w, in_channels)
                        padded_grad_input[b, h_start:h_end, w_start:w_end, :] += weight_reshaped * grad_output[b, h, w, c_out]
        
        # 去除填充
        if self.padding > 0:
            grad_input = padded_grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            grad_input = padded_grad_input
        
        return grad_input

# --------------------------- 池化层 ---------------------------
class MaxPool2D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None
        self.mask = None
    
    def forward(self, x):
        self.input = x
        batch_size, in_height, in_width, in_channels = x.shape
        
        # 计算输出形状
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        # 初始化输出和掩码
        output = np.zeros((batch_size, out_height, out_width, in_channels))
        self.mask = np.zeros_like(x, dtype=bool)
        
        # 执行最大池化
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(in_channels):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # 提取感受野
                        receptive_field = x[b, h_start:h_end, w_start:w_end, c]
                        
                        # 找到最大值位置
                        max_val = np.max(receptive_field)
                        max_index = np.unravel_index(np.argmax(receptive_field), receptive_field.shape)
                        
                        # 保存输出和掩码
                        output[b, h, w, c] = max_val
                        self.mask[b, h_start + max_index[0], w_start + max_index[1], c] = True
        
        return output
    
    def backward(self, grad_output):
        batch_size, out_height, out_width, out_channels = grad_output.shape
        grad_input = np.zeros_like(self.input)
        
        # 计算梯度
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(out_channels):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # 将梯度分配到最大值位置
                        grad_input[b, h_start:h_end, w_start:w_end, c][self.mask[b, h_start:h_end, w_start:w_end, c]] = grad_output[b, h, w, c]
        
        return grad_input

# --------------------------- 全连接层 ---------------------------
class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # 初始化权重和偏置
        variance = 2.0 / input_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(variance)
        self.bias = np.zeros(output_size)
        
        # 存储梯度
        self.grad_weights = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        # 展平输入
        if x.ndim > 2:
            batch_size = x.shape[0]
            x_flat = x.reshape(batch_size, -1)
        else:
            x_flat = x
        
        return np.dot(x_flat, self.weights) + self.bias
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        
        # 计算权重梯度
        if self.input.ndim > 2:
            x_flat = self.input.reshape(batch_size, -1)
        else:
            x_flat = self.input
        
        self.grad_weights = np.dot(x_flat.T, grad_output)
        
        # 计算偏置梯度
        self.grad_bias = np.sum(grad_output, axis=0)
        
        # 计算输入梯度
        grad_input = np.dot(grad_output, self.weights.T)
        
        # 恢复原始形状
        if self.input.ndim > 2:
            grad_input = grad_input.reshape(self.input.shape)
        
        return grad_input

# --------------------------- CNN模型 ---------------------------
class CNN:
    def __init__(self, input_shape, num_classes):
        self.layers = []
        self.loss_function = CrossEntropyLoss()
        
        # 构建模型
        self.layers.append(Conv2D(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        self.layers.append(Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        # 计算全连接层输入大小
        _, h, w, c = self._get_output_shape(input_shape)
        fc_input_size = h * w * c
        print(f"全连接层输入大小: {fc_input_size} = {h} * {w} * {c}")
        
        self.layers.append(Dense(input_size=fc_input_size, output_size=256))
        self.layers.append(ReLU())
        self.layers.append(Dense(input_size=256, output_size=128))
        self.layers.append(ReLU())
        self.layers.append(Dense(input_size=128, output_size=num_classes))
        self.layers.append(Softmax())
    
    def _get_output_shape(self, input_shape):
        # 辅助函数：计算模型中间层输出形状
        x = np.zeros(input_shape)
        
        # 计算到第二个池化层后的输出形状
        for layer in self.layers[:6]:  # 计算到第二个池化层
            x = layer.forward(x)
        
        return x.shape
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                layer.weights -= learning_rate * layer.grad_weights
                layer.bias -= learning_rate * layer.grad_bias
    
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, learning_rate=0.01):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size
        
        print(f"开始训练，共 {epochs} 轮，每轮 {num_batches} 批次")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 随机打乱训练数据
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            correct = 0
            
            for batch in range(num_batches):
                print(f"正在执行轮次 {epoch+1}/{epochs}，批次 {batch+1}/{num_batches}")
                
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                
                # 获取批次数据
                x_batch = x_train_shuffled[batch_start:batch_end]
                y_batch = y_train_shuffled[batch_start:batch_end]
                print(f"批次数据形状: x_batch={x_batch.shape}, y_batch={y_batch.shape}")
                
                # 前向传播
                print("执行前向传播...")
                y_pred = self.forward(x_batch)
                print(f"前向传播完成，y_pred形状: {y_pred.shape}")
                
                # 计算损失
                print("计算损失...")
                loss = self.loss_function.forward(y_pred, y_batch)
                epoch_loss += loss
                print(f"损失计算完成，损失值: {loss:.4f}")
                
                # 计算准确率
                print("计算准确率...")
                predictions = np.argmax(y_pred, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct += np.sum(predictions == labels)
                print(f"准确率计算完成，当前批次准确率: {np.sum(predictions == labels)/batch_size:.4f}")
                
                # 反向传播
                print("执行反向传播...")
                grad_output = self.loss_function.backward()
                self.backward(grad_output)
                print("反向传播完成")
                
                # 更新权重
                print("更新权重...")
                self.update(learning_rate)
                print("权重更新完成")
            
            # 计算轮次平均损失和准确率
            avg_train_loss = epoch_loss / num_batches
            train_accuracy = correct / num_samples
            
            # 评估测试集
            print("评估测试集...")
            test_loss, test_accuracy = self.evaluate(x_test, y_test, batch_size=batch_size)
            print(f"测试集评估完成")
            
            # 保存结果
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            
            print(f"轮次 {epoch+1}/{epochs} 完成")
            print(f"训练损失: {avg_train_loss:.4f}，训练准确率: {train_accuracy:.4f}")
            print(f"测试损失: {test_loss:.4f}，测试准确率: {test_accuracy:.4f}")
            print(f"耗时: {epoch_time:.2f} 秒\n")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    def evaluate(self, x, y, batch_size=64):
        num_samples = x.shape[0]
        num_batches = num_samples // batch_size
        
        total_loss = 0.0
        correct = 0
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            
            x_batch = x[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            
            y_pred = self.forward(x_batch)
            loss = self.loss_function.forward(y_pred, y_batch)
            total_loss += loss
            
            predictions = np.argmax(y_pred, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == labels)
        
        # 处理剩余样本
        if num_samples % batch_size != 0:
            x_batch = x[num_batches * batch_size:]
            y_batch = y[num_batches * batch_size:]
            
            y_pred = self.forward(x_batch)
            loss = self.loss_function.forward(y_pred, y_batch)
            total_loss += loss
            
            predictions = np.argmax(y_pred, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == labels)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / num_samples
        
        return avg_loss, accuracy
    
    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

# --------------------------- 数据处理 ---------------------------
def generate_synthetic_mnist():
    """生成合成的MNIST数据用于测试"""
    print("生成合成MNIST数据...")
    # 生成随机图像数据
    x_train = np.random.rand(1000, 28, 28).astype(np.float32)
    x_test = np.random.rand(100, 28, 28).astype(np.float32)
    
    # 生成随机标签
    y_train = np.random.randint(0, 10, size=1000)
    y_test = np.random.randint(0, 10, size=100)
    
    return (x_train, y_train), (x_test, y_test)

def load_mnist_from_url():
    """从网络加载MNIST数据集"""
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    local_file = 'mnist.npz'
    
    try:
        if not os.path.exists(local_file):
            print(f"正在下载MNIST数据集...")
            urllib.request.urlretrieve(url, local_file)
            print("下载完成")
        
        with np.load(local_file, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"下载数据集失败: {e}")
        print("使用合成数据进行测试...")
        return generate_synthetic_mnist()

def load_and_preprocess_mnist(use_small_data=False, small_data_size=1000):
    print("正在加载MNIST数据集...")
    start_time = time.time()
    
    # 从URL下载MNIST数据集
    (x_train, y_train), (x_test, y_test) = load_mnist_from_url()
    
    end_time = time.time()
    print(f"数据集加载完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"训练集大小: {x_train.shape[0]}，测试集大小: {x_test.shape[0]}")
    
    # 使用部分数据进行快速测试
    if use_small_data:
        x_train = x_train[:small_data_size]
        y_train = y_train[:small_data_size]
        x_test = x_test[:small_data_size // 10]
        y_test = y_test[:small_data_size // 10]
        print(f"使用小数据集训练，训练集大小: {x_train.shape[0]}，测试集大小: {x_test.shape[0]}")
    
    # 数据预处理
    print("正在预处理数据...")
    
    # 归一化到[0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # 添加通道维度 (N, H, W) -> (N, H, W, C)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # 标签独热编码
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y]
    
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    
    print("数据预处理完成")
    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
    print(f"训练标签形状: {y_train.shape}, 测试标签形状: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

# --------------------------- 可视化功能 ---------------------------
def plot_training_results(results):
    # 设置Matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    epochs = range(1, len(results['train_losses']) + 1)
    print(f"训练轮次: {epochs}")
    print(f"训练损失: {results['train_losses']}")
    print(f"测试损失: {results['test_losses']}")
    print(f"训练准确率: {results['train_accuracies']}")
    print(f"测试准确率: {results['test_accuracies']}")
    
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_losses'], 'b-', linewidth=2, label='训练损失')
    plt.plot(epochs, results['test_losses'], 'r-', linewidth=2, label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)  # 添加网格线
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_accuracies'], 'b-', linewidth=2, label='训练准确率')
    plt.plot(epochs, results['test_accuracies'], 'r-', linewidth=2, label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)  # 添加网格线
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("训练结果图像已保存为 training_results.png")
    plt.show()

def visualize_predictions(model, x_test, y_test, num_samples=10):
    # 设置Matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        image = x_test[idx].squeeze()
        label = np.argmax(y_test[idx])
        prediction = model.predict(x_test[idx:idx+1])[0]
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image, cmap='gray')
        
        if prediction == label:
            color = 'green'
        else:
            color = 'red'
        
        plt.title(f"预测: {prediction}\n真实: {label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

# --------------------------- 主函数 ---------------------------
def main():
    # 加载数据（使用小数据集进行快速测试）
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist(use_small_data=True, small_data_size=1000)
    
    # 创建模型
    input_shape = (1, 28, 28, 1)  # (batch_size, height, width, channels)
    model = CNN(input_shape=input_shape, num_classes=10)
    
    # 训练模型
    print("开始训练模型...")
    start_time = time.time()
    
    results = model.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=20,
        batch_size=32,
        learning_rate=0.01
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"训练完成，总耗时: {total_time:.2f} 秒")
    
    # 可视化结果
    print("可视化训练结果...")
    plot_training_results(results)
    
    print("可视化预测结果...")
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    # 最终评估
    print("\n最终模型评估:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()