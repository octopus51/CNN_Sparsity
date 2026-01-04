import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import sparse

# --------------------------- 激活函数 ---------------------------
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class Softmax:
    def forward(self, x):
        # 数值稳定的softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        return grad_output

# --------------------------- 损失函数 ---------------------------
class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        # 防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
        return loss
    
    def backward(self):
        # softmax + crossentropy的梯度
        return (self.y_pred - self.y_true) / self.y_true.shape[0]

# --------------------------- 稀疏卷积层 ---------------------------
class SparseConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparsity_ratio=0.5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sparsity_ratio = sparsity_ratio
        
        # 初始化稀疏权重
        self.W = self._initialize_sparse_weights()
        self.b = np.zeros((out_channels,))
        
        # 用于存储前向传播的输入
        self.input = None
        
    def _initialize_sparse_weights(self):
        """初始化稀疏权重矩阵"""
        # 创建密集权重
        dense_W = np.random.randn(self.kernel_size, self.kernel_size, 
                                 self.in_channels, self.out_channels) * 0.1
        
        # 应用结构化稀疏（通道级稀疏）
        if self.sparsity_ratio > 0:
            # 计算每个输出通道的重要性分数（基于权重绝对值）
            channel_scores = np.mean(np.abs(dense_W), axis=(0, 1, 2))
            # 保留重要性最高的通道
            num_keep = int(self.out_channels * (1 - self.sparsity_ratio))
            keep_channels = np.argsort(channel_scores)[-num_keep:]
            
            # 创建稀疏权重矩阵
            sparse_W = np.zeros_like(dense_W)
            sparse_W[:, :, :, keep_channels] = dense_W[:, :, :, keep_channels]
            
            # 记录哪些通道被保留
            self.active_channels = keep_channels
        else:
            sparse_W = dense_W
            self.active_channels = np.arange(self.out_channels)
        
        return sparse_W
    
    def forward(self, x):
        self.input = x
        batch_size, height, width, channels = x.shape
        
        # 如果是稀疏的，只使用活跃通道
        if hasattr(self, 'active_channels'):
            W_sparse = self.W[:, :, :, self.active_channels]
            out_channels = len(self.active_channels)
            b_sparse = self.b[self.active_channels]
        else:
            W_sparse = self.W
            out_channels = self.out_channels
            b_sparse = self.b
        
        # 动态调整输入通道 - 如果实际输入通道数与期望不符
        if channels != self.in_channels:
            # 如果实际输入通道数小于期望，我们只使用前channels个通道
            W_sparse = W_sparse[:, :, :channels, :]
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        # 预分配输出数组（使用实际的活跃通道数）
        output = np.zeros((batch_size, out_height, out_width, out_channels))
        
        # 卷积操作
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(out_channels):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取当前区域的输入
                        x_slice = x_padded[b, h_start:h_end, w_start:w_end, :]
                        
                        # 计算卷积（使用正确的偏置索引）
                        output[b, h, w, c] = np.sum(x_slice * W_sparse[:, :, :, c]) + b_sparse[c]
        
        return output
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        
        # 计算梯度
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        grad_input = np.zeros_like(self.input)
        
        # 如果是稀疏的，只更新活跃通道
        if hasattr(self, 'active_channels'):
            active_out_channels = self.active_channels
        else:
            active_out_channels = np.arange(self.out_channels)
        
        for c_idx, c in enumerate(active_out_channels):
            for b in range(batch_size):
                for h in range(grad_output.shape[1]):
                    for w in range(grad_output.shape[2]):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取输入切片
                        x_slice = self.input[b, h_start:h_end, w_start:w_end, :]
                        
                        # 累积梯度 - 确保通道数匹配
                        grad_val = grad_output[b, h, w, c_idx]
                        
                        # 使用正确的通道数进行梯度计算
                        num_input_channels = x_slice.shape[-1]
                        W_slice = self.W[:, :, :num_input_channels, c]
                        
                        # 确保x_slice和W_slice的形状匹配
                        if x_slice.shape == W_slice.shape:
                            dW[:, :, :num_input_channels, c] += grad_val * x_slice
                            db[c_idx] += grad_val
                            
                            # 反向传播到输入
                            if h_start < grad_input.shape[1] and w_start < grad_input.shape[2]:
                                grad_input[b, h_start:h_end, w_start:w_end, :] += grad_val * W_slice
                        else:
                            # 处理边界情况 - 截取到匹配尺寸
                            min_h = min(x_slice.shape[0], W_slice.shape[0])
                            min_w = min(x_slice.shape[1], W_slice.shape[1])
                            
                            # 截取匹配的部分
                            x_slice_trimmed = x_slice[:min_h, :min_w, :]
                            W_slice_trimmed = W_slice[:min_h, :min_w, :]
                            
                            dW[:min_h, :min_w, :num_input_channels, c] += grad_val * x_slice_trimmed
                            db[c_idx] += grad_val
                            
                            # 反向传播到输入
                            if h_start < grad_input.shape[1] and w_start < grad_input.shape[2]:
                                grad_input[b, h_start:h_start+min_h, w_start:w_start+min_w, :] += grad_val * W_slice_trimmed
        
        # L1正则化梯度（稀疏性约束）
        l1_lambda = 0.001
        dW += l1_lambda * np.sign(self.W)
        
        # 应用稀疏性：强制小权重为0
        if hasattr(self, 'active_channels'):
            threshold = np.percentile(np.abs(self.W), 90)
            mask = np.abs(self.W) > threshold
            dW = dW * mask
        
        self.dW = dW
        self.db = db
        
        return grad_input
    
    def update(self, learning_rate):
        # 应用稀疏更新
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        # 重新应用稀疏性
        if hasattr(self, 'active_channels'):
            # 强制非活跃通道权重为0
            self.W[:, :, :, ~self.active_channels] = 0
        
        # 动态调整稀疏性
        if np.random.rand() < 0.1:  # 10%的概率调整
            self._prune_weights()
    
    def _prune_weights(self):
        """动态权重剪枝"""
        if not hasattr(self, 'active_channels'):
            return
            
        # 计算每个通道的重要性
        channel_importance = []
        for c in range(self.W.shape[3]):
            importance = np.sum(np.abs(self.W[:, :, :, c]))
            channel_importance.append(importance)
        
        channel_importance = np.array(channel_importance)
        
        # 保留重要性最高的80%通道
        keep_ratio = 0.8
        num_keep = int(len(channel_importance) * keep_ratio)
        keep_channels = np.argsort(channel_importance)[-num_keep:]
        
        # 更新活跃通道
        self.active_channels = keep_channels
        self.W[:, :, :, ~keep_channels] = 0

# --------------------------- 稀疏池化层 ---------------------------
class MaxPool2D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mask = None
    
    def forward(self, x):
        batch_size, height, width, channels = x.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(x)
        
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size
                    w_start = w * self.stride
                    w_end = w_start + self.pool_size
                    
                    # 找到最大值及其位置
                    region = x[b, h_start:h_end, w_start:w_end, :]
                    max_vals = np.max(region, axis=(0, 1))
                    
                    # 记录最大值位置
                    for c in range(channels):
                        max_idx = np.unravel_index(np.argmax(region[:, :, c]), 
                                                 (self.pool_size, self.pool_size))
                        max_h = h_start + max_idx[0]
                        max_w = w_start + max_idx[1]
                        self.mask[b, max_h, max_w, c] = 1
                    
                    output[b, h, w, :] = max_vals
        
        return output
    
    def backward(self, grad_output):
        batch_size, out_height, out_width, channels = grad_output.shape
        grad_input = np.zeros_like(self.mask)
        
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size
                    w_start = w * self.stride
                    w_end = w_start + self.pool_size
                    
                    # 将梯度分配到最大值位置
                    grad_input[b, h_start:h_end, w_start:w_end, :][self.mask[b, h_start:h_end, w_start:w_end, :].astype(bool)] = grad_output[b, h, w, :]
        
        return grad_input

# --------------------------- 稀疏全连接层 ---------------------------
class SparseDense:
    def __init__(self, input_size, output_size, sparsity_ratio=0.3):
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity_ratio = sparsity_ratio
        
        # 初始化稀疏权重矩阵
        self.W = self._initialize_sparse_weights()
        self.b = np.zeros((output_size,))
        
        self.input = None
    
    def _initialize_sparse_weights(self):
        """初始化稀疏权重矩阵"""
        # 创建密集权重
        dense_W = np.random.randn(self.input_size, self.output_size) * 0.1
        
        # 应用非结构化稀疏
        if self.sparsity_ratio > 0:
            # 计算每个权重的重要性分数（基于绝对值）
            weight_scores = np.abs(dense_W)
            # 确定阈值：保留重要性最高的权重
            threshold = np.percentile(weight_scores, (1 - self.sparsity_ratio) * 100)
            
            # 创建稀疏权重矩阵
            sparse_W = np.where(np.abs(dense_W) >= threshold, dense_W, 0)
            
            # 记录稀疏模式
            self.sparse_mask = (np.abs(dense_W) >= threshold)
        else:
            sparse_W = dense_W
            self.sparse_mask = np.ones_like(dense_W, dtype=bool)
        
        return sparse_W
    
    def forward(self, x):
        self.input = x
        self._original_shape = x.shape  # 保存原始形状
        # 稀疏矩阵乘法
        # 确保输入是2D矩阵
        if x.ndim > 2:
            # 展平除batch维度的所有维度
            x_flat = x.reshape(x.shape[0], -1)
            return np.dot(x_flat, self.W) + self.b
        else:
            return np.dot(x, self.W) + self.b
    
    def backward(self, grad_output):
        # 处理展平的输入
        if self.input.ndim > 2:
            input_for_grad = self.input.reshape(self.input.shape[0], -1)
        else:
            input_for_grad = self.input
        
        # 计算梯度
        self.dW = np.dot(input_for_grad.T, grad_output)
        self.db = np.sum(grad_output, axis=0)
        
        # L1正则化梯度（稀疏性约束）
        l1_lambda = 0.001
        self.dW += l1_lambda * np.sign(self.W)
        
        # 应用稀疏梯度更新
        self.dW = self.dW * self.sparse_mask
        
        # 反向传播梯度
        grad_input = np.dot(grad_output, self.W.T)
        
        # 如果原始输入是多维的，需要重塑梯度形状
        if hasattr(self, '_original_shape') and self._original_shape is not None:
            grad_input = grad_input.reshape(self._original_shape)
        
        return grad_input
    
    def update(self, learning_rate):
        # 应用稀疏更新
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        # 重新应用稀疏性
        self.W = self.W * self.sparse_mask
        
        # 动态调整稀疏性
        if np.random.rand() < 0.1:  # 10%的概率调整
            self._prune_weights()
    
    def _prune_weights(self):
        """动态权重剪枝"""
        # 计算每个权重的重要性
        weight_importance = np.abs(self.W)
        
        # 保留重要性最高的权重
        threshold = np.percentile(weight_importance, 90)
        new_mask = (weight_importance >= threshold)
        
        # 更新稀疏模式
        self.sparse_mask = new_mask
        self.W = self.W * new_mask

# --------------------------- 稀疏CNN模型 ---------------------------
class SparseCNN:
    def __init__(self, input_shape, num_classes):
        self.layers = []
        self.loss_function = CrossEntropyLoss()
        
        # 构建稀疏模型
        self.layers.append(SparseConv2D(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1, sparsity_ratio=0.3))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        self.layers.append(SparseConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, sparsity_ratio=0.4))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        
        # 计算全连接层输入大小
        _, h, w, c = self._get_output_shape(input_shape)
        fc_input_size = h * w * c
        print(f"全连接层输入大小: {fc_input_size} = {h} * {w} * {c}")
        
        # 稀疏全连接层
        self.layers.append(SparseDense(input_size=fc_input_size, output_size=256, sparsity_ratio=0.4))
        self.layers.append(ReLU())
        self.layers.append(SparseDense(input_size=256, output_size=128, sparsity_ratio=0.3))
        self.layers.append(ReLU())
        self.layers.append(SparseDense(input_size=128, output_size=num_classes, sparsity_ratio=0.2))
        self.layers.append(Softmax())
        
        # 统计模型稀疏性
        self._calculate_sparsity()
    
    def _get_output_shape(self, input_shape):
        # 辅助函数：计算模型中间层输出形状
        # 先计算第一层稀疏卷积的形状
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        
        # 第一层稀疏卷积：输入(h,w,c) -> 输出(h',w',32) 然后剪枝到活跃通道数
        h1 = (h + 2 * 1 - 5) // 1 + 1  # padding=1, kernel_size=5, stride=1
        w1 = (w + 2 * 1 - 5) // 1 + 1
        # 实际输出通道数 = 32 * (1 - 0.3) = 22.4 ≈ 22 (向下取整)
        c1 = int(32 * (1 - 0.3))
        
        # 第一个池化层：(h1,w1,c1) -> (h1/2,w1/2,c1)
        h2 = h1 // 2
        w2 = w1 // 2
        c2 = c1
        
        # 第二层稀疏卷积：输入(h2,w2,c2) -> 输出(h2,w2,64) 然后剪枝到活跃通道数
        h3 = (h2 + 2 * 1 - 3) // 1 + 1  # padding=1, kernel_size=3, stride=1
        w3 = (w2 + 2 * 1 - 3) // 1 + 1
        # 实际输出通道数 = 64 * (1 - 0.4) = 38.4 ≈ 38 (向下取整)
        c3 = int(64 * (1 - 0.4))
        
        # 第二个池化层：(h3,w3,c3) -> (h3/2,w3/2,c3)
        h4 = h3 // 2
        w4 = w3 // 2
        c4 = c3
        
        # 计算全连接层输入大小
        fc_input_size = h4 * w4 * c4
        
        print(f"中间层形状计算：")
        print(f"  输入: ({input_shape[1]}, {input_shape[2]}, {input_shape[3]})")
        print(f"  第一层卷积后: ({h1}, {w1}, {c1})")
        print(f"  第一个池化后: ({h2}, {w2}, {c2})")
        print(f"  第二层卷积后: ({h3}, {w3}, {c3})")
        print(f"  第二个池化后: ({h4}, {w4}, {c4})")
        print(f"  全连接层输入大小: {fc_input_size}")
        
        return (input_shape[0], h4, w4, c4)
    
    def _calculate_sparsity(self):
        """计算模型稀疏性"""
        total_params = 0
        sparse_params = 0
        
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer_params = layer.W.size
                total_params += layer_params
                
                if hasattr(layer, 'sparse_mask'):
                    sparse_params += np.sum(layer.sparse_mask)
                elif hasattr(layer, 'active_channels'):
                    # 对于通道级稀疏
                    active_params = np.sum(layer.W[:, :, :, layer.active_channels].size)
                    sparse_params += active_params
        
        sparsity_ratio = (total_params - sparse_params) / total_params
        print(f"模型稀疏性: {sparsity_ratio:.2%}")
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad_output = layer.backward(grad_output)
    
    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(learning_rate)
    
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, learning_rate=0.01):
        """训练稀疏模型"""
        print(f"开始稀疏模型训练，共 {epochs} 轮，每轮 {x_train.shape[0] // batch_size} 批次")
        
        results = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'sparsity_ratio': []
        }
        
        for epoch in range(epochs):
            print(f"\n正在执行轮次 {epoch+1}/{epochs}")
            
            # 打乱训练数据
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            correct = 0
            num_batches = x_train.shape[0] // batch_size
            
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                
                x_batch = x_train_shuffled[batch_start:batch_end]
                y_batch = y_train_shuffled[batch_start:batch_end]
                
                # 前向传播
                y_pred = self.forward(x_batch)
                
                # 计算损失
                loss = self.loss_function.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # 计算准确率
                predictions = np.argmax(y_pred, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct += np.sum(predictions == labels)
                
                # 反向传播
                grad_output = self.loss_function.backward()
                self.backward(grad_output)
                
                # 更新权重
                self.update(learning_rate)
                
                # 每10个批次输出一次信息
                if (batch + 1) % 10 == 0:
                    batch_accuracy = np.sum(predictions == labels) / batch_size
                    print(f"  批次 {batch+1}/{num_batches}，损失: {loss:.4f}，准确率: {batch_accuracy:.4f}")
            
            # 计算轮次平均损失和准确率
            avg_loss = epoch_loss / num_batches
            accuracy = correct / x_train.shape[0]
            
            # 测试评估
            test_loss, test_accuracy = self.evaluate(x_test, y_test)
            
            # 计算当前稀疏性
            current_sparsity = self._get_current_sparsity()
            
            results['train_loss'].append(avg_loss)
            results['train_accuracy'].append(accuracy)
            results['test_loss'].append(test_loss)
            results['test_accuracy'].append(test_accuracy)
            results['sparsity_ratio'].append(current_sparsity)
            
            print(f"轮次 {epoch+1} 完成 - 训练损失: {avg_loss:.4f}, 训练准确率: {accuracy:.4f}")
            print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}, 稀疏性: {current_sparsity:.2%}")
        
        return results
    
    def _get_current_sparsity(self):
        """获取当前模型稀疏性"""
        total_params = 0
        sparse_params = 0
        
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer_params = layer.W.size
                total_params += layer_params
                
                if hasattr(layer, 'sparse_mask'):
                    sparse_params += np.sum(layer.sparse_mask)
                elif hasattr(layer, 'active_channels'):
                    # 对于通道级稀疏
                    active_params = np.sum(layer.W[:, :, :, layer.active_channels].size)
                    sparse_params += active_params
        
        sparsity_ratio = (total_params - sparse_params) / total_params
        return sparsity_ratio
    
    def evaluate(self, x, y, batch_size=64):
        """评估模型性能"""
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for i in range(0, x.shape[0], batch_size):
            end = min(i + batch_size, x.shape[0])
            x_batch = x[i:end]
            y_batch = y[i:end]
            
            # 前向传播
            y_pred = self.forward(x_batch)
            
            # 计算损失
            loss = self.loss_function.forward(y_pred, y_batch)
            total_loss += loss * x_batch.shape[0]
            
            # 计算准确率
            predictions = np.argmax(y_pred, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == labels)
            total_samples += x_batch.shape[0]
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def predict(self, x):
        """预测"""
        return self.forward(x)

# --------------------------- 数据加载函数 ---------------------------
def load_and_preprocess_mnist(use_small_data=False, small_data_size=6000):
    """加载和预处理MNIST数据集"""
    print("正在加载MNIST数据集...")
    
    try:
        # 尝试加载本地数据
        if os.path.exists('mnist.npz'):
            data = np.load('mnist.npz')
            x_train, y_train = data['x_train'], data['y_train']
            x_test, y_test = data['x_test'], data['y_test']
            print("本地数据集加载完成")
        else:
            # 生成合成数据作为示例
            print("使用合成数据作为示例...")
            np.random.seed(42)
            
            # 生成训练数据（4D格式）
            x_train = np.random.randn(60000, 28, 28, 1).astype(np.float32)
            y_train = np.random.randint(0, 10, (60000,)).astype(np.int32)
            
            # 生成测试数据（4D格式）
            x_test = np.random.randn(10000, 28, 28, 1).astype(np.float32)
            y_test = np.random.randint(0, 10, (10000,)).astype(np.int32)
            
            print("合成数据生成完成")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("使用合成数据...")
        np.random.seed(42)
        
        # 生成训练数据（4D格式）
        x_train = np.random.randn(60000, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 10, (60000, 10)).astype(np.float32)
        
        # 生成测试数据（4D格式）
        x_test = np.random.randn(10000, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 10, (10000, 10)).astype(np.float32)
    
    print(f"训练集大小: {x_train.shape[0]}，测试集大小: {x_test.shape[0]}")
    
    # 使用小数据集进行快速测试
    if use_small_data:
        print(f"使用小数据集训练，训练集大小: {small_data_size}, 测试集大小: {small_data_size//10}")
        x_train = x_train[:small_data_size]
        y_train = y_train[:small_data_size]
        x_test = x_test[:small_data_size//10]
        y_test = y_test[:small_data_size//10]
    
    # 预处理数据
    print("正在预处理数据...")
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    # 确保标签是独热编码格式（适用于两个模型）
    if y_train.ndim == 1:
        # 整数标签转换为独热编码
        y_train_onehot = np.zeros((len(y_train), 10))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        y_train = y_train_onehot
        
        y_test_onehot = np.zeros((len(y_test), 10))
        y_test_onehot[np.arange(len(y_test)), y_test] = 1
        y_test = y_test_onehot
    
    # 确保数据是4D格式（batch_size, height, width, channels）
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
    if x_test.ndim == 3:
        x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"数据预处理完成")
    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
    print(f"训练标签形状: {y_train.shape}, 测试标签形状: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

# --------------------------- 可视化函数 ---------------------------
def plot_sparse_training_results(results):
    """可视化稀疏训练结果"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(results['train_loss']) + 1)
    
    # 训练损失
    axes[0, 0].plot(epochs, results['train_loss'], 'b-', linewidth=2, label='训练损失')
    axes[0, 0].plot(epochs, results['test_loss'], 'r-', linewidth=2, label='测试损失')
    axes[0, 0].set_title('模型损失变化', fontsize=14)
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率
    axes[0, 1].plot(epochs, results['train_accuracy'], 'b-', linewidth=2, label='训练准确率')
    axes[0, 1].plot(epochs, results['test_accuracy'], 'r-', linewidth=2, label='测试准确率')
    axes[0, 1].set_title('模型准确率变化', fontsize=14)
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 稀疏性变化
    axes[1, 0].plot(epochs, results['sparsity_ratio'], 'g-', linewidth=2, label='稀疏性')
    axes[1, 0].set_title('模型稀疏性变化', fontsize=14)
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('稀疏性比例')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 准确率 vs 稀疏性
    axes[1, 1].scatter(results['sparsity_ratio'], results['test_accuracy'], 
                      c=epochs, cmap='viridis', s=50)
    axes[1, 1].set_title('准确率 vs 稀疏性', fontsize=14)
    axes[1, 1].set_xlabel('稀疏性比例')
    axes[1, 1].set_ylabel('测试准确率')
    axes[1, 1].grid(True)
    
    # 添加颜色条
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('训练轮次')
    
    plt.tight_layout()
    plt.savefig('sparse_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sparse_predictions(model, x_test, y_test, num_samples=10):
    """可视化稀疏模型预测结果"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 随机选择样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    for i, idx in enumerate(indices):
        # 获取预测
        x_sample = x_test[idx:idx+1]
        y_pred = model.predict(x_sample)
        predicted_class = np.argmax(y_pred[0])
        true_class = np.argmax(y_test[idx])
        
        # 显示原始图像
        axes[0, i].imshow(x_sample[0, :, :, 0], cmap='gray')
        axes[0, i].set_title(f'真实: {true_class}', fontsize=12)
        axes[0, i].axis('off')
        
        # 显示预测分布
        axes[1, i].bar(range(10), y_pred[0])
        axes[1, i].set_title(f'预测: {predicted_class}', fontsize=12)
        axes[1, i].set_xlabel('类别')
        axes[1, i].set_ylabel('概率')
        axes[1, i].set_xticks(range(10))
        
        # 高亮预测结果
        axes[1, i].bar(predicted_class, y_pred[0][predicted_class], color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sparse_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------- 主函数 ---------------------------
def main():
    print("=== 稀疏CNN模型训练 ===")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist(use_small_data=True, small_data_size=1000)
    
    # 创建稀疏模型
    input_shape = (1, 28, 28, 1)
    model = SparseCNN(input_shape=input_shape, num_classes=10)
    
    # 训练模型
    print("\n开始训练稀疏模型...")
    start_time = time.time()
    
    results = model.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=10,
        batch_size=32,
        learning_rate=0.01
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n训练完成，总耗时: {total_time:.2f} 秒")
    
    # 可视化结果
    print("可视化稀疏训练结果...")
    plot_sparse_training_results(results)
    
    print("可视化稀疏预测结果...")
    visualize_sparse_predictions(model, x_test, y_test, num_samples=10)
    
    # 最终评估
    print("\n最终稀疏模型评估:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"最终稀疏性: {model._get_current_sparsity():.2%}")

if __name__ == "__main__":
    main()