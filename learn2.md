## 前后端数据交互分析

### 一、整体通信架构

前端与后端通过**两种通道**交互：

| 通道 | 技术 | 用途 |
|------|------|------|
| **HTTP REST** | `fetch` → Flask Blueprint `/api/*` | 数据集保存/列表/删除、训练启动/状态、推理启动/步进/停止、模型管理 |
| **WebSocket** | Socket.IO (polling) → Flask-SocketIO | 实时小车动作下发、状态回传、ACT推理 |

---

### 二、信息采集流程

采集分为 **前端本地采集** 和 **后端共享采集** 两条并行路径：

#### 1. 前端本地模拟 + 采集（`SimPage.tsx`）

这是核心采集路径，**所有模拟和采集都在前端完成，不需要后端实时参与**。

**渲染循环**（20 FPS，`692:763`）：

```
requestAnimationFrame → 50ms 节流
  ├─ ACT推理（5Hz，每200ms一次 POST /api/infer/step）
  ├─ updatePhysics()（物理更新：键盘→指令→移动→碰撞检测）
  ├─ drawTopDown()（俯视图Canvas渲染）
  ├─ drawFirstPerson()（伪3D第一人称Canvas渲染）
  └─ 数据采集（如果 collecting=true）
```

**每帧采集的数据**（`743:762`）：

```typescript
EpisodeStep = {
    state: number[];     // 14维 — 小车自身状态
    envState: number[];  // 7维  — 环境观测状态
    action: number[];    // 5维  — one-hot 动作编码
    image?: string;      // base64 PNG — 第一人称画布截图
}
```

**`getObservationVectors()`**（`496:515`）构建观测向量：

- **state（14维）**：`[x, y, angle, speed, ballDistance, 0,0,0,0,0,0,0,0,0]` — 只用了前5维
- **envState（7维）**：`[x, y, angle, speed, isCollision, forwardDistance, ballDistance]`

其中 `forwardDistance` 通过射线-线段求交（`465:479`）计算前方障碍物距离，`ballDistance` 计算到最近圆形目标的距离。

**动作编码**（`450:463`）：

| 指令 | one-hot向量 |
|------|------------|
| up | `[1,0,0,0,0]` |
| down | `[0,1,0,0,0]` |
| left | `[0,0,1,0,0]` |
| right | `[0,0,0,1,0]` |
| stop | `[0,0,0,0,5]` |

**采样策略**（`746:751`）：
- 非stop指令：每帧都记录
- stop指令：每5帧记录1次（`STOP_RECORD_STRIDE=5`），避免冗余数据

**回合管理**（`828:858`）：
- 每回合至少15步（`MIN_STEPS_PER_EPISODE=15`）
- 用户点击"复位保存" → 当前回合存入 `episodesRef` → 小车重置到地图中心
- 达到目标回合数（默认8轮）后自动结束采集

#### 2. 前端 → 后端的数据保存

`packDataset()`（`517:556`）将所有回合数据打包为 chunk 结构：

```
每10步为一个chunk（CHUNK_SIZE=10）
  ├─ state: 取chunk第一步的state（1×14）
  ├─ env_state: 取chunk第一步的envState（1×7）
  ├─ action: chunk内所有步的动作序列（10×5）
  ├─ action_is_pad: padding标记（不足10步补0，padding位置为1）
  └─ images: 每步的第一人称截图 base64（10×1）
```

通过 `POST /api/dataset` 发送到后端，后端（`api.py:65:185`）：
1. 验证 JSON payload 完整性
2. 将 states/env_states/actions/action_is_pad 转为 `torch.Tensor`
3. 解码 base64 图片 → 保存为 PNG 文件到 `output/datasets/images_xxx/` 目录
4. 记录图片相对路径到 dataset 的 `obs_images` 字段
5. 整体序列化为 `act_dataset_xxx.pt`（PyTorch 文件）

---

### 三、训练流程

#### 1. 触发训练

前端 `startTrain()`（`239:260`）→ `POST /api/train/start` → 后端启动**独立线程**

后端 `_run_training()`（`287:336`）：
```
train_thread (daemon=True)
  └─ train_from_dataset()
       ├─ load_dataset_from_local(path)  → torch.load → ACTDataset
       ├─ build_config(state_dim=14, env_state_dim=7, action_dim=5, chunk_size=10)
       ├─ ACT(config)  → 创建 Transformer 模型
       └─ train_act()  → 训练循环
```

#### 2. 训练细节（`train.py:44:174`）

**模型架构**（ACT - Action Chunking Transformer）：
- 输入：state（14维）+ env_state（7维）→ 线性投影 → Transformer Encoder
- 输出：Transformer Decoder → 生成 **chunk_size 步** 的动作序列（10×5）
- 可选 VAE：encoder 将动作编码为 latent → decoder 重建（本场景默认关闭）

**训练循环**：
```
每 epoch:
  每 batch:
    1. model(batch) → actions_pred（B,10,5）, (mu, log_sigma)
    2. loss = MSE(actions_pred, actions_gt)  // 行为克隆
    3. 可选: loss += kl_weight * KL_loss     // VAE正则
    4. loss.backward() → grad_clip(1.0) → optimizer.step()
    5. progress_callback(epoch, total, avg_loss) → 更新 train_state（线程安全）
```

**前端轮询训练状态**（`222:237`）：每秒 `GET /api/train/status` 获取 epoch、loss、progress

#### 3. 训练产物

保存到 `output/train/act_YYYYMMDD_HHMMSS/`：
- `act_checkpoint.pt` — 包含 `model_state_dict` + `state_dim` + `env_state_dim` + `action_dim` + `chunk_size`
- `training_metrics.json` — 每步的 loss、lr、epoch 等指标

---

### 四、推理流程

#### 1. 加载模型

前端选择模型 → `POST /api/infer/start` → 后端（`api.py:490:528`）：
1. `torch.load(checkpoint)` → 读取维度参数
2. `build_config()` + `ACT(config)` + `load_state_dict()` → 恢复模型
3. 存入 `infer_state`（内存中常驻）

#### 2. 步进推理

渲染循环中 5Hz 触发（`701:738`）：
```
每 200ms:
  1. getObservationVectors() → 获取当前 {state:14维, env_state:7维}
  2. POST /api/infer/step {state, env_state}
  3. 后端: tensor(state) → model({obs_state, env_state}) → actions[0,0] → _map_action()
  4. _map_action(): 取5维向量中最大值索引 → "up"/"down"/"left"/"right"/"stop"
  5. _apply_action(): 更新后端 car 物理状态 + socketio.emit('car_state')
  6. 返回 {status:"ok", action:"up"} → 前端 actCommandRef.current = "up"
  7. updatePhysics() 读取 actCommandRef → 驱动前端小车移动
```

关键点：推理时**后端也同步驱动了 car 单例**（`_apply_action`），并通过 WebSocket 广播状态，但前端主要用返回的 action 字符串在本地 `applyLocalAction()` 驱动渲染。

---

### 五、数据流全景图

```
┌──────────────── 前端（SimPage.tsx）────────────────┐
│                                                     │
│  键盘WASD ──→ 指令 ──→ applyLocalAction()          │
│       │                ├─ 物理更新(本地carState)     │
│       │                ├─ 碰撞检测                   │
│       │                └─ emit('action') ──────────→│──→ Socket.IO ──→ 后端 car 单例
│       │                                               │
│  采集时:                                              │
│  每帧 getObservationVectors() → {state, envState}     │
│  commandToActionVec(cmd) → [one-hot]                 │
│  fpvCanvas.toDataURL() → base64 PNG                  │
│  → EpisodeStep 存入 currentEpisode                    │
│                                                     │
│  复位保存 → packDataset() → POST /api/dataset ──────→│──→ HTTP ──→ 后端保存 .pt + PNG
│                                                     │
│  开始训练 → POST /api/train/start ─────────────────→│──→ HTTP ──→ 后端训练线程
│  轮询状态 → GET /api/train/status ←─────────────────│←── HTTP ←── train_state
│                                                     │
│  ACT推理(5Hz):                                       │
│  state+envState → POST /api/infer/step ────────────→│──→ HTTP ──→ 后端 model()
│  ←── {action:"up"} ←───────────────────────────────│←── HTTP ←── action映射
│  actCommandRef = "up" → applyLocalAction("up")      │
│                                                     │
│  俯视图 Canvas          第一人称 Canvas              │
│  (800×600 2D)          (320×240 伪3D)               │
└─────────────────────────────────────────────────────┘
```

**核心设计思想**：前端完全自治模拟物理和渲染，后端只负责三件事 —— 数据持久化（`.pt`）、模型训练（ACT Transformer）、模型推理（tensor forward pass）。前端通过 HTTP API 与后端交互，两者是松耦合的。