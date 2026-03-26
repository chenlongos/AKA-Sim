# AKA-Sim 阿卡0号开源机器人

AKA-Sim 是一个用于模拟环境训练和 ACT (Action Chunking Transformer) 策略学习的平台。

## 功能特性

- **模拟环境** - 内置 3D 小车物理模拟器
- **ACT 策略学习** - 基于 LeRobot 实现的 Action Chunking Transformer
- **Web 控制界面** - html前端 + Flask 后端，支持实时控制
- **WebSocket 通信** - 实时双向通信，低延迟控制

## 技术栈

| 类别 | 技术                                        |
|------|-------------------------------------------|
| 后端 | Flask, Flask-SocketIO, PyTorch            |
| 前端 | HTML |
| 机器学习 | PyTorch, einops, torchvision              |
| 实时通信 | Socket.IO                                 |

## 目录结构

```
AKA-Sim/
├── app/                    # Flask 应用
│   └── routes/            # API 路由
├── backend/               # 核心源代码
│   ├── app/               # Flask 应用模块
│   │   └── routes/       # API 路由
│   ├── cameras/          # 摄像头接口
│   │   └── opencv/       # OpenCV 摄像头
│   ├── configs/          # 配置文件
│   ├── policies/act/     # ACT 神经网络模型
│   ├── sim/model/        # 小车物理模型
│   ├── train.py          # ACT 训练逻辑
│   └── utils/            # 工具函数
├── frontend/              # React 前端
├── main.html             # HTML 前端
├── docs/                 # 项目文档
├── run.py                # 主入口
└── requirements.txt      # Python 依赖
```

## 快速开始

### 方式一：Docker（推荐）

项目提供了多阶段 Docker 镜像，包含开发环境和生产环境两种构建目标。远程镜像已发布，可直接拉取使用，无需本地构建。

#### 环境要求

- Docker 20.10+
- Docker Compose 2.0+（可选）

#### 1. 拉取预构建镜像（推荐）

跳过构建，直接拉取已有镜像：

```bash
docker pull docker.cnb.cool/tmacychen/chenlongos-aka-sim:latest
docker tag docker.cnb.cool/tmacychen/chenlongos-aka-sim:latest aka-sim:dev
```

#### 2. 启动容器

```bash
# 启动开发容器（挂载本地源码）
docker run -d --name aka-sim-dev \
  -p 80:80 -p 5000:5000 \
  -v $(pwd):/app \
  -v /app/frontend/node_modules \
  -e PYTHONPATH=/app/backend \
  -w /app \
  aka-sim:dev

# 首次启动需要构建前端静态资源
docker exec aka-sim-dev bash -c "cd /app/frontend && npm run build"
```

访问 http://localhost:80 即可使用。

#### 3. 本地构建镜像

如需自定义构建，可使用以下命令：

```bash
# 构建开发镜像
docker build --target dev -t aka-sim:dev .

# 构建生产镜像
docker build --target prod -t aka-sim:prod .

# 启动开发容器（挂载本地源码，支持热重载）
docker run -d --name aka-sim-dev \
  -p 80:80 -p 5000:5000 \
  -v $(pwd):/app \
  -v /app/frontend/node_modules \
  -e PYTHONPATH=/app/backend \
  -w /app \
  aka-sim:dev

# 首次启动需要构建前端静态资源
docker exec aka-sim-dev bash -c "cd /app/frontend && npm run build"

# 启动生产容器
docker run -d --name aka-sim-prod \
  -p 80:80 -p 443:443 \
  -e PYTHONPATH=/app/backend \
  aka-sim:prod
```

#### 使用 Docker Compose

```bash
# 开发环境（源码挂载，支持热重载）
docker compose up dev

# 生产环境（构建前端，仅运行时依赖）
docker compose up prod --build
```

#### 端口说明

| 端口 | 用途 | 说明 |
|------|------|------|
| 80 | Flask HTTP | 主服务端口 |
| 443 | Flask HTTPS | 生产环境，自动生成自签名证书 |
| 5000 | Flask HTTP 备用 | Windows 环境下的默认端口 |
| 5173 | Vite 开发服务器 | 仅开发环境 |

#### Dockerfile 构建阶段

| 阶段 | 说明 | 基础镜像 |
|------|------|----------|
| `base` | 安装 Python 3.11、Node.js 20 及所有依赖 | python:3.11-slim |
| `dev` | 开发环境，额外安装 debugpy/watchdog/flake8/black | base |
| `prod` | 生产环境，自动构建 React 前端并复制到 static/ | base |

#### GPU 支持（可选）

如需 CUDA 加速，将 Dockerfile 基础镜像替换为：
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base
```
并安装对应版本的 PyTorch，运行时添加 `--gpus all` 参数。

### 方式二：本地安装

#### 1. 环境配置

创建 Python 3.11 环境：

```bash
conda create -n aka-sim python=3.11 -y
conda activate aka-sim
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 构建前端（React）

```bash
cd frontend && npm ci && npm run build && cd ..
```

#### 4. 启动后端服务

```bash
PYTHONPATH=backend python run.py
```

#### 5. 访问服务

- http://localhost/ - 首页
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

## 相关链接

- [LeRobot](https://github.com/huggingface/lerobot) - ACT 策略实现参考

## License

MIT
