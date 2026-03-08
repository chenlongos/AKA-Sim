# AKA-Sim 阿卡0号开源机器人

AKA-00 是一个功能完整的开源机器人项目，支持实体机器人控制、模拟环境训练和 ACT (Action Chunking Transformer) 策略学习。

## 功能特性

- **模拟环境** - 内置 3D 小车物理模拟器
- **ACT 策略学习** - 基于 LeRobot 实现的 Action Chunking Transformer
- **Web 控制界面** - React + html前端 + Flask 后端，支持实时控制
- **WebSocket 通信** - 实时双向通信，低延迟控制

## 技术栈

| 类别 | 技术                                       |
|------|------------------------------------------|
| 后端 | Flask, Flask-SocketIO, PyTorch           |
| 前端 | React 19, TypeScript, Vite, Zustand, html |
| 机器学习 | PyTorch, einops, torchvision             |
| 实时通信 | Socket.IO                                |

## 目录结构

```
AKA-Sim/
├── app/                    # Flask 应用
│   └── routes/            # API 路由
├── background/            # 核心源代码
│   ├── train.py           # ACT 训练逻辑
│   ├── arm_control/       # 舵机控制
│   ├── cameras/           # 摄像头接口
│   ├── policies/act/      # ACT 神经网络模型
│   └── sim/model/         # 小车物理模型
├── frontend/              # React 前端
├── main.html              # html前端
├── docs/                  # 项目文档
├── run.py                 # 主入口
└── requirements.txt       # Python 依赖
```

## 快速开始

### 1. 环境配置

创建 Python 3.11 环境：

```bash
conda create -n aka-sim python=3.11 -y
conda activate aka-sim
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动后端服务

```bash
python run.py
```

后端启动后，在浏览器中打开前端页面：

```
http://localhost/main.html
```

服务启动后访问：
- http://localhost/main.html - 主控制界面
- http://localhost/sim - 模拟器

## API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/control` | GET | 小车控制 |
| `/api/ip` | GET | 获取IP地址 |
| `/api/dataset` | POST | 创建数据集 |
| `/api/datasets` | GET | 获取数据集列表 |
| `/api/datasets/<id>` | DELETE | 删除数据集 |
| `/api/train/start` | POST | 开始训练 |
| `/api/train/status` | GET | 训练状态 |
| `/api/models` | GET | 获取模型列表 |
| `/api/models/<id>` | DELETE | 删除模型 |
| `/api/infer/start` | POST | 开始推理 |
| `/api/infer/step` | POST | 推理步进 |
| `/api/infer/stop` | POST | 停止推理 |
| `/ws` | WebSocket | 实时通信 |

## 硬件支持

- **开发板**: OPI5P (SG2002)
- **舵机**: STS3215, MG996R, ZL, ZP10S
- **摄像头**: OpenCV 支持的 USB 摄像头

## 文档

更多详细信息请参阅 [docs/](docs/) 目录：

- [快速开始](docs/src/start.md) - 项目启动指南
- [初始化配置](docs/src/init.md) - 硬件和网络配置
- [开发文档](docs/dev/) - 开发相关文档

## 相关链接

- [LeRobot](https://github.com/huggingface/lerobot) - ACT 策略实现参考

## License

MIT
