# 后端训练说明文档

## 1. 训练原理

后端训练是指将数据上传到云端服务器，利用高性能计算资源 (GPU) 进行模型训练。这种方式适合处理大规模数据集和复杂的模型架构。

### 核心机制
- **模型架构**：基于 ACT (Action Chunking with Transformers) 模型。
- **输入数据**：
  - **图像**：64x64 像素的 RGB 图像，来自小车的车载摄像头。
  - **状态向量**：14 维向量，包含小车的位置、速度、角度、目标距离等信息。
  - **动作向量**：5 维向量，包含线速度、角速度以及其他控制信号。
- **输出数据**：预测未来 `CHUNK_SIZE` (默认为 10) 个时间步的动作序列。
- **损失函数**：均方误差 (Mean Squared Error, MSE) + KL 散度 (KL Divergence)，用于优化动作预测和隐空间分布。
- **优化器**：AdamW 优化器。

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
在上传到后端前，前端会将数据打包成 JSON 格式：
- **episodes**: 包含所有回合的数据。
- **metadata**: 包含机器人类型、动作空间、FPS 等信息。

后端接收到数据后，会进行以下处理：
- **归一化**：对图像进行归一化处理 (0-1)。
- **填充**：如果回合长度不足 `CHUNK_SIZE`，会进行填充。
- **批处理**：将数据分批次送入模型训练。

## 3. 实现方法

### 代码位置
- `src/services/cloudService.ts`: 负责与后端 API 交互，上传数据和启动训练。
- `src/App.tsx`: 负责数据采集和调用云端服务。

### 关键函数
1.  **`packDataset(episodes)`**:
    -   将前端收集的回合数据打包成后端可接受的格式。
    -   提取 `state` (14维) 和 `envState` (7维)。
    -   提取 `action` (5维) 和 `action_is_pad` (用于标记填充数据)。
    -   提取 `image` (Base64 编码或像素数组)。

2.  **`saveCloudDataset(dataset)`**:
    -   调用 `cloudService.uploadDataset` 将打包好的数据上传到云端存储。

3.  **`startTraining(datasetPath, modelId)`**:
    -   调用 `cloudService.startTraining` 通知后端开始训练。
    -   后端会启动一个新的训练任务，并返回任务 ID。

### 训练流程
1.  用户在界面上操作小车，录制多个回合的数据。
2.  点击“Save Dataset (Cloud)”按钮。
3.  `App.tsx` 调用 `packDataset` 打包数据。
4.  `cloudService.uploadDataset` 上传数据。
5.  用户选择已上传的数据集和模型配置。
6.  点击“Start Training (Cloud)”按钮。
7.  后端接收请求，启动训练进程。
8.  训练完成后，模型会自动保存，并可用于云端推理。
