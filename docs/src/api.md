# AKA-Sim API 文档

AKA-Sim 机器人的前端 API 文档，包含 REST API 和 WebSocket 两种接口。

**Base URL**: `http://localhost:5000`

---

## 目录

- [REST API](#rest-api)
  - [数据集管理](#数据集管理)
  - [模型管理](#模型管理)
  - [训练接口](#训练接口)
  - [推理接口](#推理接口)
  - [远程控制](#远程控制)
- [WebSocket API](#websocket-api)
  - [小车控制事件](#小车控制事件)
  - [ACT 推理事件](#act-推理事件)

---

## REST API

### 数据集管理

#### 1. 获取数据集列表

```
GET /api/datasets
```

**响应示例**:
```json
{
  "datasets": [
    {
      "id": "act_dataset_20240315_143022.pt",
      "path": "/path/to/dataset.pt",
      "size_bytes": 1234567,
      "created_at": "2024-03-15 14:30:22",
      "updated_at": "2024-03-15 14:30:22"
    }
  ]
}
```

---

#### 2. 保存数据集

```
POST /api/dataset
```

**请求体**:
```json
{
  "states": [[...], [...]],
  "env_states": [[...], [...]],
  "actions": [[...], [...]],
  "action_is_pad": [[false], [false]],
  "rewards": [1.0, 1.0],
  "dones": [false, true],
  "truncateds": [false, false],
  "images": [[["data:image/png;base64,..."]]]
}
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `states` | number[][] | 是 | 小车状态数据 |
| `env_states` | number[][] | 是 | 环境状态数据 |
| `actions` | number[][] | 是 | 动作数据 |
| `action_is_pad` | boolean[][] | 是 | 动作填充标记 |
| `rewards` | number[] | 否 | 奖励数据 |
| `dones` | boolean[] | 否 | 完成标记 |
| `truncateds` | boolean[] | 否 | 截断标记 |
| `images` | string[][][] | 否 | 图像数据 (Base64) |
| `meta` | object | 否 | 元数据 |

**响应示例**:
```json
{
  "status": "success",
  "path": "/path/to/save/act_dataset_20240315_143022.pt"
}
```

---

#### 3. 删除数据集

```
DELETE /api/datasets/{dataset_id}
```

**路径参数**:
- `dataset_id`: 数据集文件名

**响应示例**:
```json
{
  "status": "deleted",
  "dataset_id": "act_dataset_20240315_143022.pt",
  "removed_image_dir": true
}
```

---

### 模型管理

#### 4. 获取模型列表

```
GET /api/models
```

**响应示例**:
```json
{
  "models": [
    {
      "id": "act_20240315_143022",
      "path": "/path/to/checkpoint/act_checkpoint.pt",
      "created_at": "2024-03-15 14:30:22",
      "updated_at": "2024-03-15 14:30:22"
    }
  ]
}
```

---

#### 5. 删除模型

```
DELETE /api/models/{model_id}
```

**路径参数**:
- `model_id`: 模型目录名

**响应示例**:
```json
{
  "status": "deleted",
  "model_id": "act_20240315_143022"
}
```

---

### 训练接口

#### 6. 开始训练

```
POST /api/train/start
```

**请求体**:
```json
{
  "dataset_path": "/path/to/dataset.pt",
  "num_epochs": 50,
  "batch_size": 64,
  "lr": 0.0003,
  "device": "cuda",
  "use_vae": false,
  "kl_weight": 1.0,
  "grad_clip_norm": 1.0
}
```

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `dataset_path` | string | 否 | 最新数据集 | 数据集路径 |
| `num_epochs` | number | 否 | 50 | 训练轮数 |
| `batch_size` | number | 否 | 64 | 批次大小 |
| `lr` | number | 否 | 3e-4 | 学习率 |
| `device` | string | 否 | cuda/cpu | 设备 |
| `use_vae` | boolean | 否 | false | 是否使用 VAE |
| `kl_weight` | number | 否 | 1.0 | KL 权重 |
| `grad_clip_norm` | number | 否 | 1.0 | 梯度裁剪 |

**响应示例**:
```json
{
  "status": "started",
  "run_id": "20240315_143022",
  "dataset_path": "/path/to/dataset.pt"
}
```

---

#### 7. 获取训练状态

```
GET /api/train/status
```

**响应示例**:
```json
{
  "status": "running",
  "run_id": "20240315_143022",
  "dataset_path": "/path/to/dataset.pt",
  "model_id": "act_20240315_143022",
  "epoch": 25,
  "num_epochs": 50,
  "avg_loss": 0.123,
  "progress": 0.5,
  "message": null,
  "error": null,
  "started_at": "2024-03-15 14:30:22",
  "ended_at": null,
  "updated_at": "2024-03-15 14:35:22"
}
```

**状态说明**:
- `idle`: 空闲
- `starting`: 启动中
- `running`: 训练中
- `completed`: 完成
- `failed`: 失败

---

### 推理接口

#### 8. 启动推理

```
POST /api/infer/start
```

**请求体**:
```json
{
  "model_id": "act_20240315_143022",
  "model_path": "/path/to/checkpoint.pt",
  "device": "cuda"
}
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `model_id` | string | 否 | 模型 ID (与 model_path 二选一) |
| `model_path` | string | 否 | 模型路径 |
| `device` | string | 否 | 设备 (cuda/cpu) |

**响应示例**:
```json
{
  "status": "started",
  "model_id": "act_20240315_143022"
}
```

---

#### 9. 执行推理步骤

```
POST /api/infer/step
```

**请求体**:
```json
{
  "state": [0.1, 0.2, 0.5, 0.0],
  "env_state": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `state` | number[] | 否 | 小车状态 (默认使用当前状态) |
| `env_state` | number[] | 否 | 环境状态 (默认全零) |

**响应示例**:
```json
{
  "status": "ok",
  "action": "up",
  "action_vector": [0.85, 0.12, -0.03, 0.01, 0.05]
}
```

**动作说明**:
- `up`: 前进
- `down`: 后退
- `left`: 左转
- `right`: 右转
- `stop`: 停止

---

#### 10. 停止推理

```
POST /api/infer/stop
```

**响应示例**:
```json
{
  "status": "stopped"
}
```

---

### 远程控制

#### 11. 获取本机 IP

```
GET /api/ip
```

**响应示例**:
```json
{
  "ip": "192.168.1.100"
}
```

---

#### 12. 发送远程控制指令

```
GET /api/control?action=up&speed=50&time=0&target_ip=192.168.1.101
```

**查询参数**:
| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `action` | string | 是 | stop | 动作 (up/down/left/right/stop) |
| `speed` | number | 是 | 50 | 速度 |
| `time` | number | 是 | 0 | 持续时间 (秒) |
| `target_ip` | string | 是 | - | 目标机器人 IP |

**响应示例**:
```json
{
  "status": "ok",
  "response": "..."
}
```

**错误响应**:
```json
{
  "error": "missing target_ip"
}
```

---

## WebSocket API

连接地址: `http://localhost:5000`

**事件命名空间**: `/` (默认)

---

### 小车控制事件

#### 13. 发送动作指令

**客户端发送**:
```javascript
socket.emit('action', 'up');
```

**可用动作**: `up`, `down`, `left`, `right`, `stop`

**服务端响应**:
```javascript
socket.on('car_state', (state) => {
  console.log(state);
});
```

**状态数据格式**:
```json
{
  "x": 0.0,
  "y": 0.0,
  "angle": 0.0,
  "speed": 0.0,
  "maxSpeed": 100.0,
  "acceleration": 5.0,
  "friction": 0.95,
  "rotationSpeed": 0.1
}
```

---

#### 14. 获取小车状态

**客户端发送**:
```javascript
socket.emit('get_car_state');
```

**服务端响应**:
```javascript
socket.on('car_state', (state) => {
  // 小车状态数据
});
```

---

#### 15. 重置小车状态

**客户端发送**:
```javascript
socket.emit('reset_car_state');
```

**服务端响应**:
```javascript
socket.on('car_state', (state) => {
  // 重置后的状态
});
```

---

### ACT 推理事件

#### 16. ACT 模型推理

**客户端发送**:
```javascript
socket.emit('act_infer', {
  observation: {
    state: [0.1, 0.2, 0.5, 0.0],
    environment_state: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
});
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `observation.state` | number[] | 是 | 小车状态 |
| `observation.environment_state` | number[] | 是 | 环境状态 |

**服务端响应**:
```javascript
socket.on('act_action', (data) => {
  console.log(data.action);  // [[0.85, 0.12, ...]]
});
```

**错误响应**:
```javascript
{
  "error": "missing observation"
}
```

---

## 前端使用示例

### REST API 调用

```typescript
// 获取数据集列表
const response = await fetch('/api/datasets');
const data = await response.json();

// 开始训练
await fetch('/api/train/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ num_epochs: 50 })
});
```

### WebSocket 调用

```typescript
import { io } from 'socket.io-client';

const socket = io('http://localhost:5000');

// 发送动作
socket.emit('action', 'up');

// 监听状态
socket.on('car_state', (state) => {
  console.log('Car state:', state);
});

// ACT 推理
socket.emit('act_infer', {
  observation: {
    state: [0.1, 0.2, 0.5, 0.0],
    environment_state: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
});

socket.on('act_action', (data) => {
  console.log('Action:', data.action);
});
```

---

## 错误码说明

| 状态码 | 描述 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 503 | 连接失败 |
| 504 | 请求超时 |
