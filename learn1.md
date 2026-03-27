## AKA-Sim (阿卡0号开源机器人) — 项目分析与学习指南

---

### 一、项目是什么

这是一个面向 **AKA-00 机器人小车** 的开源模拟训练平台，核心目标是实现 **ACT (Action Chunking Transformer)** 策略学习。用户可以在浏览器中操控虚拟小车、采集数据、训练神经网络、让模型自主驾驶。

---

### 二、技术栈总览

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.11 + Flask + Flask-SocketIO |
| 机器学习 | PyTorch, einops, torchvision, accelerate |
| ACT 模型 | 自实现的 Action Chunking Transformer (~500行) |
| React 前端 | React 19 + TypeScript + Vite 8 + Zustand |
| HTML 前端 | Three.js 3D + TensorFlow.js (单文件) |
| 部署 | Docker 多阶段构建 + Docker Compose |

---

### 三、项目架构

```
/workspace/
├── run.py                    # 主入口，启动 Flask HTTP/HTTPS
├── backend/
│   ├── app/
│   │   ├── __init__.py       # Flask 工厂函数
│   │   ├── extensions.py     # SocketIO 实例
│   │   └── routes/
│   │       ├── api.py        # REST API (数据集/模型/训练/推理/控制) ~640行
│   │       ├── frontend.py   # 前端路由 SPA fallback
│   │       └── websocket.py  # WebSocket 事件处理
│   ├── policies/act/
│   │   ├── configuration_act.py  # ACT 配置
│   │   └── modeling_act.py       # ACT Transformer 完整实现 ~500行
│   ├── sim/model/car.py      # 简单车体物理模型
│   ├── train.py              # 训练流程 ~390行
│   ├── configs/types.py      # 策略特征类型定义
│   └── utils/                # 常量 & 随机工具
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── BaseControlPage.tsx  # 远程控制台
│       │   └── SimPage.tsx          # 2D 模拟器 ~1050行 (核心)
│       ├── components/target/       # 目标物管理组件
│       ├── api/socket.ts            # Socket.IO + fetch 封装
│       └── store/targetStore.ts     # Zustand 状态管理
├── main.html                 # 独立 Three.js 前端 ~1500行
├── docker-compose.yml        # dev/prod 两种部署
└── Dockerfile                # 多阶段构建
```

---

### 四、推荐学习路径

#### 阶段 1：理解项目目标与整体架构 (1-2天)

| 顺序 | 内容 | 文件 |
|------|------|------|
| 1 | 阅读项目文档 | `docs/src/index.md`, `docs/src/start.md` |
| 2 | 阅读入口与配置 | `run.py`, `backend/app/__init__.py` |
| 3 | 了解 API 接口 | `docs/src/api.md`, `backend/app/routes/api.py` |
| 4 | 理解通信机制 | `backend/app/routes/websocket.py`, `frontend/src/api/socket.ts` |

**目标**: 理解前后端如何通过 REST API + WebSocket 协作。

#### 阶段 2：后端核心 — ACT 模型与训练 (3-5天)

| 顺序 | 内容 | 文件 |
|------|------|------|
| 1 | 理解模型配置 | `backend/policies/act/configuration_act.py` |
| 2 | 深入 ACT 模型实现 | `backend/policies/act/modeling_act.py` |
| 3 | 理解数据集与训练流程 | `backend/train.py` |
| 4 | 理解物理模型 | `backend/sim/model/car.py` |
| 5 | 了解常量定义 | `backend/utils/constants.py` |

**关键知识点**:
- **ACT 架构**: Transformer encoder-decoder + 可选 VAE encoder + 可选 ResNet18 视觉骨干
- **训练流程**: 采集数据 → `.pt` 文件 → 后台线程训练 → 保存 checkpoint
- **推理流程**: 加载模型 → 5Hz 推理循环 → 输出动作向量

> **前置知识**: PyTorch 基础、Transformer 架构、VAE、LeRobot 数据格式

#### 阶段 3：前端核心 — 2D 模拟器 (3-5天)

| 顺序 | 内容 | 文件 |
|------|------|------|
| 1 | 路由与入口 | `frontend/src/main.tsx`, `frontend/src/App.tsx` |
| 2 | 远程控制台 | `frontend/src/pages/BaseControlPage.tsx` |
| 3 | **2D 模拟器 (核心)** | `frontend/src/pages/SimPage.tsx` |
| 4 | 目标物渲染引擎 | `frontend/src/components/target/TargetRenderer.tsx` |
| 5 | 状态管理 | `frontend/src/store/targetStore.ts`, `frontend/src/model/target.ts` |

**关键知识点**:
- Canvas 2D 渲染 (俯视图 + 伪3D第一人称射线投射)
- 20FPS 游戏循环 (requestAnimationFrame)
- 数据采集流程：开始 → 操控 → 复位保存 → 上传
- ACT 自主推理的 5Hz 循环

#### 阶段 4：独立 3D 前端 (进阶，2-3天)

| 内容 | 文件 |
|------|------|
| Three.js 3D 渲染 + 多场景 | `main.html` |
| 前端 TensorFlow.js 训练 | `main.html` 中的 ACT 模型 |
| 车载摄像头渲染 | WebGLRenderTarget |

#### 阶段 5：部署与实战 (1-2天)

| 内容 | 文件 |
|------|------|
| Docker 部署 | `Dockerfile`, `docker-compose.yml` |
| 真机部署 | `init.sh`, `https_init.sh` |

---

### 五、动手实践建议

1. **先跑起来**:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm ci && npm run build && cd ..
   python run.py
   ```
   打开浏览器访问，手动操控小车，体验数据采集 → 训练 → 推理的完整流程。

2. **阅读顺序建议**:
   - `api.py` → 理解数据如何存储和查询
   - `train.py` → 理解如何从数据集训练模型
   - `modeling_act.py` → 深入理解 ACT Transformer 实现
   - `SimPage.tsx` → 理解前端如何将一切串联起来

3. **推荐前置知识**:
   - Python Flask Web 开发
   - PyTorch 深度学习基础
   - Transformer / VAE 架构原理
   - React + TypeScript 基础
   - WebSocket 实时通信
   - Canvas 2D 绘图基础

---

### 六、核心数据流

```
手动操控采集数据 → POST /api/dataset (保存 .pt)
       ↓
POST /api/train/start → 后台 PyTorch 训练 ACT
       ↓
POST /api/infer/start → POST /api/infer/step (5Hz)
       ↓
模型输出动作 → 小车自动行驶
```

总结：这个项目是一个非常好的 **端到端机器人学习** 实战项目，涵盖了数据采集、模型训练、在线推理、Web 可视化等完整链路，适合对具身智能/机器人学习方向感兴趣的开发者深入学习。



## 前后端 REST API + WebSocket 协作分析

### 项目概览

AKA-Sim 是一个面向 AKA-00 机器人小车的模拟训练平台，实现了 ACT (Action Chunking Transformer) 策略学习。核心数据流是：**手动控制采集数据 → 训练 ACT 模型 → 模型自主推理**。

---

### 架构总览

```
┌──────────────────────────────────────────────────┐
│               单一 Flask 容器                      │
│                                                  │
│  React SPA (/:80) ──┐                            │
│  Three.js  (/:80) ──┤                            │
│                      ▼                            │
│  ┌─── REST API (/api/*) ────┐                    │
│  │  数据集 CRUD              │                    │
│  │  训练 启动/状态查询        │  ←─ fetch 轮询    │
│  │  推理 启动/步骤/停止       │  ←─ 5Hz POST      │
│  │  远程控制代理              │                    │
│  └──────────────────────────┘                    │
│                                                  │
│  ┌─── Socket.IO (/socket.io) ─┐                  │
│  │  action         C → S      │  ←─ 20FPS 实时   │
│  │  get_car_state  C → S      │                    │
│  │  reset_car_state C → S    │                    │
│  │  car_state      S → C      │  ←─ 状态广播      │
│  │  act_infer      C → S      │  ←─ WS推理(备用)  │
│  │  act_action     S → C      │                    │
│  └──────────────────────────┘                    │
│                                                  │
│  PyTorch ACT Model (线程内)                       │
│  CarModel 单例 (全局状态)                          │
└──────────────────────────────────────────────────┘
```

---

### 通信职责划分

**REST API** — 用于**数据密集型、非实时**操作：

| 端点 | 方法 | 用途 | 调用频率 |
|------|------|------|---------|
| `/api/dataset` | POST | 保存采集数据集 | 采集结束时 1 次 |
| `/api/datasets` | GET | 列出数据集 | 初始化 + 采集后刷新 |
| `/api/datasets/<id>` | DELETE | 删除数据集 | 手动 |
| `/api/models` | GET | 列出模型 | 初始化 + 手动 |
| `/api/models/<id>` | DELETE | 删除模型 | 手动 |
| `/api/train/start` | POST | 启动训练 | 按钮触发 |
| `/api/train/status` | GET | 查询训练进度 | **1 秒轮询** |
| `/api/infer/start` | POST | 加载推理模型 | 按钮触发 |
| `/api/infer/step` | POST | 执行一步推理 | **5Hz 轮询** |
| `/api/infer/stop` | POST | 停止推理 | 按钮触发 |
| `/api/control` | GET | 远程控制代理 | 实车模式 |
| `/api/ip` | GET | 获取本机 IP | 初始化 |

**Socket.IO** — 用于**高频实时**操作：

| 事件 | 方向 | 用途 | 频率 |
|------|------|------|------|
| `action` | C→S | 发送控制指令 (up/down/left/right/stop) | **20FPS** |
| `car_state` | S→C | 广播小车位置/角度 | 每次 action 后 |
| `get_car_state` | C→S | 查询当前状态 | 初始化时 1 次 |
| `reset_car_state` | C→S | 重置小车 | 复位按钮 |
| `act_infer` / `act_action` | C↔S | ACT 推理 (备用通道) | 按需 |

> **注意**: Socket.IO 配置为 `transports: ["polling"]`, `upgrade: false`，实际使用 HTTP 长轮询而非原生 WebSocket。

---

### 核心业务流程

#### 1. 数据采集 (手动控制)

```
用户 WASD 键盘
  → socket.emit('action', 'up')           [20FPS]
  → 后端 handle_action() 更新 CarModel
  → socket.emit('car_state', {x,y,angle})
  → 前端更新画布坐标

同时前端本地采集:
  → getObservationVectors()  [state[14], envState[7]]
  → fpvCanvas.toDataURL()    [base64 截图]
  → push 到 currentEpisodeRef
```

#### 2. 数据集保存

```
采集结束 → packDataset(episodes)
  → POST /api/dataset  {states, env_states, actions, action_is_pad, images}
  → 后端 base64 解码图片 → torch.save() → output/datasets/*.pt
```

#### 3. 模型训练

```
POST /api/train/start {dataset_path, num_epochs, ...}
  → 后端 threading.Thread 后台运行 train_act()
  → 更新全局 train_state
  ← GET /api/train/status (前端 1s 轮询)
  → 返回 {status, epoch, avg_loss, progress}
```

#### 4. ACT 自主推理

```
POST /api/infer/start {model_id}          → 加载模型
POST /api/infer/step {state, env_state}   → 5Hz 推理循环
  → 后端 ACT 前向推理 → _map_action() → "up"
  → _apply_action() → emit('car_state')
  ← 返回 {action: "up", action_vector: [...]}
  → 前端 actCommandRef 控制 updatePhysics()
```

---

### 关键架构特点

1. **混合通信策略**: REST 处理数据 CRUD 和训练流程，Socket.IO 处理高频实时控制，职责清晰
2. **推理双通道**: SimPage 的 ACT 推理走 REST `POST /api/infer/step`（带状态同步），WebSocket 的 `act_infer` 是备用通道（Three.js 前端使用）
3. **无认证无数据库**: 所有端点完全开放，数据以 `.pt` 文件存储，零外部依赖
4. **坐标偏移**: 后端 CarModel 以 `(0,0)` 为中心，前端画布偏移 `(400,300)` 映射
5. **SPA 部署**: 生产环境 Flask 同时提供 API + React 静态文件，开发环境 Vite 代理 `/api` 和 `/socket.io` 到 Flask 5000 端口




## `main.html` vs React 前端对比分析

`main.html` 是一个**独立的 3D 模拟器页面**，与当前 React 前端的 `/sim` 页面功能相似但实现完全不同：

### 核心差异

| 维度 | `main.html` (Three.js) | React 前端 `/sim` (Canvas 2D) |
|------|----------------------|------------------------------|
| **3D 渲染** | **Three.js + WebGL**，真3D场景（网格体、材质、光影） | **Canvas 2D 光线投射**（Wolfenstein 3D 风格伪3D） |
| **视觉效果** | 纹理地面（木纹/瓷砖/草地）、动态光照、阴影、雾效 | 扁平色块，无纹理、无光影，像素风 |
| **场景** | 4种场景（基础/客厅/教室/网球场）+ 复杂度调节 + 光源控制 | 固定网格，自由放置矩形/圆形障碍物 |
| **相机** | 第三人称跟随相机 + 车载第一人称相机（320x240） | 俯视图 + 伪3D第一人称 |
| **物理引擎** | 简单碰撞检测，无旋转碰撞 | 支持旋转矩形碰撞、圆形碰撞 |
| **前端训练** | TensorFlow.js 在浏览器端训练 ACT 模型 | 无前端训练，仅云端训练 |
| **训练模式** | 双模式：**前端TF.js训练** + **云端训练** | 仅云端训练 |
| **推理** | 前端 TF.js 推理（Temporal Ensembling）+ 云端推理 | 仅云端推理 5Hz |
| **目标管理** | 固定目标点（红色方块 + 黄色球），无编辑器 | 完整 CRUD：创建/编辑/删除/拖拽/导入导出 |
| **数据采集** | 64x64 图像 + 14维状态向量 + one-hot action | 类似，320x240 图像作为 base64 |
| **通信** | 仅 REST API（无 WebSocket） | Socket.IO + REST API |
| **状态持久化** | 无 | Zustand + localStorage |

### 关键总结

1. **`main.html` 更"炫"** — Three.js 真正的 3D 渲染，多场景多纹理，视觉效果远超 React 版
2. **`main.html` 更完整的功能闭环** — 支持浏览器端 TF.js 训练 + 推理，不依赖后端也能完成 AI 流程
3. **React 前端更实用** — 目标编辑器更完善（拖拽/导入导出/持久化），WebSocket 实时通信，更适合实际数据采集
4. **两者互不替代** — `main.html` 像是一个早期原型或演示页面，React 前端是正式版本