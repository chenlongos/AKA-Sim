# 前端训练说明文档

## 1. 训练原理

前端训练是指在用户的浏览器中直接利用 TensorFlow.js 进行模型训练。这种方式不需要将数据上传到服务器，保护了用户隐私，并且可以利用用户本地的 GPU 加速训练。

### 核心机制
- **模型架构**：使用卷积神经网络 (CNN) 处理图像输入，结合全连接层 (Dense) 处理状态向量输入。
- **输入数据**：
  - **图像**：64x64 像素的 RGB 图像，来自小车的车载摄像头。
  - **状态向量**：14 维向量，包含小车的位置、速度、角度、目标距离等信息。
- **输出数据**：预测未来 `CHUNK_SIZE` (默认为 10) 个时间步的动作序列。每个动作包含线速度和角速度。
- **损失函数**：均方误差 (Mean Squared Error, MSE)，用于衡量预测动作与实际动作之间的差异。
- **优化器**：Adam 优化器。

## 2. 数据集定义

数据集由一系列“回合” (Episode) 组成，每个回合包含多个时间步 (Frame)。

### 单个 Frame 的结构
```json
{
  "state": [x, z, angle, speed, targetDist, ...zeros], // 14维向量
  "envState": [x, z, angle, speed, collision, forwardDist, targetDist], // 7维向量 (用于环境交互)
  "image": [[[r,g,b], ...], ...], // 64x64x3 像素矩阵
  "action": [velocity, angularVelocity, 0, 0, 0] // 动作向量 (目前主要使用前两维)
}
```

### 数据预处理
在训练前，数据会被处理成以下格式：
- **imageInputs**: `[BatchSize, 64, 64, 3]`
- **stateInputs**: `[BatchSize, 14]`
- **labelChunks**: `[BatchSize, CHUNK_SIZE * 2]` (展平的动作序列)

## 3. 实现方法

### 代码位置
- `src/services/actService.ts`: 包含模型定义、数据预处理和训练逻辑。
- `src/App.tsx`: 负责数据采集和调用训练服务。

### 关键函数
1.  **`prepareTrainingData(episodes)`**:
    -   遍历所有回合。
    -   对于每个时间步，提取当前图像和状态。
    -   提取未来 `CHUNK_SIZE` 个时间步的动作作为标签 (Label)。
    -   将数据转换为 TensorFlow.js 可接受的数组格式。

2.  **`createModel()`**:
    -   定义双输入模型：
        -   `imageInput`: 卷积层 -> 池化层 -> 展平。
        -   `stateInput`: 全连接层。
    -   拼接两个分支的输出。
    -   通过全连接层输出预测的动作序列。

3.  **`trainModel(model, data)`**:
    -   将数据转换为 `tf.Tensor`。
    -   调用 `model.fit()` 开始训练。
    -   配置 `batchSize` 为 32，`epochs` 为 50。

### 训练流程
1.  用户在界面上操作小车，录制多个回合的数据。
2.  点击“Start Training”按钮。
3.  `App.tsx` 调用 `actService.prepareTrainingData` 处理数据。
4.  `actService.createModel` 创建新模型。
5.  `actService.trainModel` 使用处理好的数据训练模型。
6.  训练完成后，模型可用于推理 (Inference)。
