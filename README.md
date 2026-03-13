# LeRobot ACT Simulator - 具身智能教学仿真平台

## 📖 项目简介

**LeRobot ACT Simulator** 是一个基于 Web 的轻量级具身智能（Embodied AI）教学仿真平台。它旨在通过可视化的方式，演示机器人如何通过**模仿学习（Imitation Learning）**掌握运动控制技能。

本项目复现了 **ACT (Action Chunking with Transformers)** 的核心思想（简化版），提供了一个完整的从**数据采集**、**模型训练**到**自主推理**的闭环流程，全部在浏览器端通过 JavaScript 实现，无需后端 GPU 服务器支持。

---

## 🏗️ 技术架构

本项目采用现代前端技术栈构建，实现了 3D 物理仿真与深度学习的无缝结合。

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **视图层** | **React 18** | 构建响应式 UI，管理应用状态 |
| **3D 仿真** | **Three.js** | 渲染机器人、环境、物理碰撞检测 |
| **深度学习** | **TensorFlow.js** | 浏览器端的模型构建、训练与推理 |
| **样式** | **Tailwind CSS** | 打造现代、科技感的界面风格 |
| **构建工具** | **Vite** | 高性能的前端构建与开发服务器 |

---

## 🧠 核心实现：从演示到智能

这是本项目的核心部分，实现了机器人"观察-学习-行动"的智能闭环。

### 1. 数据采集 (Data Collection)

为了教会机器人，我们需要先通过人工演示（Teleoperation）生成专家数据。

*   **机制**：以 10Hz 的频率记录机器人的**状态（State）**和**动作（Action）**。
*   **状态空间**：机器人位置 `(x, z)`、朝向 `(sinθ, cosθ)`、目标位置 `(tx, tz)`。
*   **动作空间**：线速度 `v`、角速度 `ω`。

```typescript
// 核心采集逻辑 (src/App.tsx)
const recordFrame = useCallback(() => {
  if (!sim.current.target || !sim.current.isRecording) return;
  
  const { robotState, target } = sim.current;
  
  // 构建当前帧数据
  const frame = {
      state: [
          robotState.x, 
          robotState.z, 
          Math.sin(robotState.rotation), // 使用三角函数编码角度，避免不连续性
          Math.cos(robotState.rotation),
          target.position.x,
          target.position.z
      ],
      action: [
          robotState.velocity,       // 记录专家操作的速度
          robotState.angularVelocity // 记录专家操作的转向
      ]
  };
  
  sim.current.currentEpisode.push(frame);
}, []);
```

### 2. 模型训练 (Model Training)

收集到数据后，我们使用 TensorFlow.js 在浏览器中训练一个神经网络。

*   **模型架构**：多层感知机 (MLP)。
    *   输入层：6 维状态向量。
    *   隐藏层：64 神经元 (ReLU) -> 32 神经元 (ReLU)。
    *   输出层：2 维动作向量 (线性激活)。
*   **训练目标**：最小化预测动作与专家动作之间的均方误差 (MSE)。

```typescript
// 训练流程实现 (src/App.tsx)
const startTraining = async () => {
  // 1. 数据预处理：将收集到的 Episodes 转换为 Tensor
  const inputs = [];
  const labels = [];
  sim.current.episodes.forEach(episode => {
      episode.forEach(frame => {
          inputs.push(frame.state);
          labels.push(frame.action);
      });
  });
  
  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(labels);
  
  // 2. 定义模型结构
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [6] }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 2, activation: 'linear' })); // 输出速度和角速度
  
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  
  // 3. 开始训练 (50 Epochs)
  await model.fit(xs, ys, {
      epochs: 50,
      callbacks: {
          onEpochEnd: (epoch, logs) => {
              // 实时更新 UI 进度条
              setTrainingProgress(((epoch + 1) / 50) * 100);
          }
      }
  });
};
```

### 3. 自主推理 (Autonomous Inference)

训练完成后，机器人接管控制权，根据当前观察到的状态自主决策。

*   **闭环控制**：每 50ms (20Hz) 进行一次推理。
*   **泛化能力**：即使目标位置随机变化，模型也能根据训练学到的策略（"靠近目标"）规划路径。

```typescript
// 实时推理循环 (src/App.tsx)
const runInference = useCallback(() => {
  const { robotState, target, model } = sim.current;
  
  // 1. 构建当前状态张量
  const input = tf.tensor2d([[
      robotState.x,
      robotState.z,
      Math.sin(robotState.rotation),
      Math.cos(robotState.rotation),
      target.position.x,
      target.position.z
  ]]);
  
  // 2. 模型预测动作
  const prediction = model.predict(input) as tf.Tensor;
  const data = prediction.dataSync(); // 获取同步数据
  
  // 3. 执行动作
  robotState.velocity = data[0];
  robotState.angularVelocity = data[1];
  
  // 4. 资源清理与循环
  input.dispose();
  prediction.dispose();
  setTimeout(runInference, 50);
}, []);
```

---

## 🌍 仿真环境细节

为了提供逼真的教学体验，我们在 Three.js 中构建了丰富的环境细节：

*   **物理引擎**：实现了简化的 AABB (Axis-Aligned Bounding Box) 碰撞检测，防止机器人穿墙。
*   **多场景支持**：支持基础场景、客厅、教室、网球场等多种环境切换。
*   **第一人称视角**：通过 `WebGLRenderTarget` 实时渲染机器人车载摄像头的画面，模拟真实的视觉输入。

---

## 🚀 快速开始

1.  **采集数据**：点击“开始采集”，使用键盘 `WASD` 控制小车移动到红色目标方块处。建议采集 3-5 条不同起点的路径。
2.  **训练模型**：点击“开始训练模型”，观察 Loss 下降曲线。
3.  **自主推理**：训练完成后，点击“启动自主推理”，见证机器人自动寻找目标！

---

*Designed with ❤️ for AI Education.*
