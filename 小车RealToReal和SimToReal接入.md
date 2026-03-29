# 小车 RealToReal 和 SimToReal 接入

这份说明面向已经有真实小车的用户。

目标是统一一套数据结构，让两类数据都能进入现有 ACT 训练链路：

- `RealToReal`: 真实小车采集，训练后继续部署到真实小车
- `SimToReal`: 仿真数据采集，训练后部署到真实小车

## 1. 新增的数据结构

后端新增了统一 schema，位置在：

- `backend/transfer/schema.py`
- `backend/transfer/converter.py`

核心对象：

- `CameraFrame`: 摄像头单帧
- `ObservationFrame`: 一次观测，包含 `state`、`environment_state`、`cameras`
- `ControlAction`: 一次控制动作，支持 `command`、`vector`、`throttle/steering`
- `TransferStep`: 一步轨迹
- `TransferEpisode`: 一段轨迹，带 `source_domain` 和 `transfer_mode`

## 2. 两种模式怎么表示

### RealToReal

```json
{
  "episode_id": "real_ep_001",
  "source_domain": "real",
  "transfer_mode": "real_to_real"
}
```

### SimToReal

```json
{
  "episode_id": "sim_ep_001",
  "source_domain": "sim",
  "transfer_mode": "sim_to_real"
}
```

## 3. 统一请求格式

通过下面接口把 episode 转成现有 ACT 数据集：

```http
POST /api/transfer/dataset
```

最小请求体示例：

```json
{
  "chunk_size": 10,
  "episodes": [
    {
      "episode_id": "real_ep_001",
      "source_domain": "real",
      "transfer_mode": "real_to_real",
      "robot_id": "car_01",
      "teleoperator_id": "operator_a",
      "steps": [
        {
          "observation": {
            "state": [0.1, 0.2, 0.0, 0.3],
            "environment_state": [0.0, 1.0, 0.0, 0.0],
            "cameras": [
              {
                "camera_id": "front",
                "data_url": "data:image/jpeg;base64,..."
              }
            ]
          },
          "action": {
            "command": "up"
          }
        }
      ]
    }
  ]
}
```

服务端会自动完成：

- 轨迹切 chunk
- `action_is_pad` 补齐
- 图像落盘
- 生成 `.pt` 数据集

## 4. 真实小车采集示例

如果你的机器已经安装了 OpenCV，可以直接使用新增封装：

- `backend/cameras/opencv/camera_opencv.py`

最小 Python 示例：

```python
import requests

from backend.cameras.opencv.camera_opencv import OpenCVCamera


with OpenCVCamera(device_index=0, width=640, height=480) as cam:
    image_data_url = cam.read_data_url()

payload = {
    "chunk_size": 10,
    "episodes": [
        {
            "episode_id": "real_ep_001",
            "source_domain": "real",
            "transfer_mode": "real_to_real",
            "robot_id": "car_01",
            "teleoperator_id": "operator_a",
            "steps": [
                {
                    "observation": {
                        "state": [0.1, 0.2, 0.0, 0.3],
                        "environment_state": [0.0, 1.0, 0.0, 0.0],
                        "cameras": [
                            {
                                "camera_id": "front",
                                "data_url": image_data_url,
                            }
                        ],
                    },
                    "action": {
                        "command": "up"
                    },
                }
            ],
        }
    ],
}

resp = requests.post("http://127.0.0.1:5000/api/transfer/dataset", json=payload, timeout=30)
print(resp.json())
```

## 5. 控制动作支持两种写法

### 方式 1：离散命令

```json
{
  "action": {
    "command": "left"
  }
}
```

默认会映射到：

- `up`
- `down`
- `left`
- `right`
- `stop`

### 方式 2：连续控制

```json
{
  "action": {
    "throttle": 0.7,
    "steering": -0.4
  }
}
```

服务端会自动离散化成 one-hot 动作向量，便于直接进入当前 ACT 训练链路。

### 方式 3：直接给动作向量

```json
{
  "action": {
    "vector": [0, 0, 1, 0, 0]
  }
}
```

如果同时给了 `vector` 和 `command`，优先使用 `vector`。

## 6. 接到现有训练链路的方法

`/api/transfer/dataset` 生成的文件，本质上还是现有 ACT 数据集，因此后续流程不变：

1. 调用 `/api/train/start`
2. 训练得到 `output/train/act_xxx/act_checkpoint.pt`
3. 调用 `/api/infer/start`
4. 循环调用 `/api/infer/step`

## 7. 现成可运行脚本

新增了两个示例脚本：

- `scripts/real_car_collect.py`
- `scripts/real_car_infer.py`

### 7.1 真实小车采集脚本

作用：

- 打开真实摄像头
- 接终端键盘收控制
- 一边控制小车，一边记录 observation/action
- 最后上传成 `RealToReal` 或 `SimToReal` 数据集

示例：

```bash
venv/bin/python scripts/real_car_collect.py \
  --backend-url http://127.0.0.1:5000 \
  --robot-base-url http://192.168.1.101 \
  --telemetry-url http://192.168.1.101:8000/observation \
  --robot-id car_01 \
  --teleoperator-id operator_a \
  --transfer-mode real_to_real
```

控制键：

- `w/a/s/d`: 前进/左转/后退/右转
- `space` 或 `x`: 停止
- `n`: 结束当前 episode，开始下一个
- `u`: 立即上传当前已完成的 episodes
- `q`: 退出并上传

### 7.2 在线视觉推理客户端

作用：

- 打开真实摄像头
- 获取实时状态观测
- 把 `state + environment_state + image` 发给 AKA-Sim 后端
- 拿到模型输出动作后发送给真实小车

示例：

```bash
venv/bin/python scripts/real_car_infer.py \
  --backend-url http://127.0.0.1:5000 \
  --robot-base-url http://192.168.1.101 \
  --telemetry-url http://192.168.1.101:8000/observation \
  --model-id act_20260321_120000
```

如果只想验证推理链路，不真的发控制，可以加：

```bash
--dry-run
```

### 7.3 telemetry 接口最小格式

脚本默认会去 `--telemetry-url` 拉一个 JSON，并读取：

```json
{
  "state": [0.1, 0.2, 0.0, 0.3],
  "environment_state": [0.0, 1.0, 0.0, 0.0]
}
```

也兼容这种嵌套写法：

```json
{
  "observation": {
    "state": [0.1, 0.2, 0.0, 0.3],
    "environment_state": [0.0, 1.0, 0.0, 0.0]
  }
}
```

如果没提供 `--telemetry-url`，脚本会用零向量占位。

## 8. 当前实现边界

已经实现的部分：

- 统一 RealToReal / SimToReal 轨迹结构
- 支持摄像头帧随轨迹一起保存
- 支持离散动作、连续控制、动作向量三种动作描述
- 服务端自动转换成当前 ACT 数据集格式
- 提供真实小车采集脚本与在线视觉推理客户端示例

当前仍然需要你自己提供的部分：

- 真实小车的底盘控制驱动
- 真实小车的状态估计
- 把真实传感器转换成 `state` 和 `environment_state`

另外要注意：

- 当前 ACT 主训练链路已经支持 `state + environment_state + images`
- `/api/infer/step` 和 WebSocket `act_infer` 也都能接收当前观测图像
- 服务端会按 checkpoint 中记录的视觉配置自动恢复相机数量与输入尺寸
