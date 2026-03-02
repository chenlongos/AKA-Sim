"""
AKA-Sim 数据导出模块 - 将采集的数据导出为ACT训练格式
"""

import base64
import json
import os
import struct
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch


def export_dataset(
    samples: List[Dict[str, Any]],
    output_dir: str = None,
    chunk_size: int = 1000,
    action_chunk_size: int = 16,
) -> str:
    """
    将采集的样本导出为ACT训练数据格式

    Args:
        samples: 采集的样本列表，每个样本包含 image, state, actions
        output_dir: 输出目录
        chunk_size: 每个parquet文件的最大样本数
        action_chunk_size: 动作分块大小

    Returns:
        导出的目录路径
    """
    # 默认保存到项目根目录
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "dataset"
    else:
        output_dir = Path(output_dir)

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_dir)
    data_dir = output_path / "data"
    videos_dir = output_path / "videos" / "observation.images.fpv"
    meta_dir = output_path / "meta" / "episodes"

    # 创建目录
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 动作编码: forward=0, backward=1, left=2, right=3, stop=4
    action_to_idx = {"forward": 0, "backward": 1, "left": 2, "right": 3, "stop": 4}

    # 状态编码 (7维)
    # [x, y, angle, speed, maxSpeed, acceleration, rotationSpeed]
    state_keys = ["x", "y", "angle", "speed", "maxSpeed", "acceleration", "rotationSpeed"]

    num_samples = len(samples)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    print(f"导出 {num_samples} 个样本到 {output_dir}")
    print(f"分为 {num_chunks} 个数据块")

    # 状态统计
    all_states = []
    action_counts = {k: 0 for k in action_to_idx.keys()}

    for i, sample in enumerate(samples):
        # 解码并保存图像
        image_data = sample["image"]
        if image_data.startswith("data:image"):
            # 移除 data:image/jpeg;base64, 前缀
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        # 保存图像文件
        chunk_idx = i // chunk_size
        file_idx = i % chunk_size
        image_filename = f"frame_{file_idx:06d}.jpg"
        chunk_dir = videos_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)
        image_path = chunk_dir / image_filename

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # 收集状态和动作统计
        state_values = [sample["state"].get(k, 0) for k in state_keys]
        all_states.append(state_values)

        for action in sample.get("actions", []):
            if action in action_counts:
                action_counts[action] += 1

        # 定期打印进度
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i + 1}/{num_samples}")

    # 计算状态归一化参数
    states_array = np.array(all_states, dtype=np.float32)

    # 确保是2维数组
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    state_mean = states_array.mean(axis=0)
    state_std = states_array.std(axis=0)
    state_min = states_array.min(axis=0)
    state_max = states_array.max(axis=0)

    # 转换为tensor避免除零问题
    state_min_tensor = torch.from_numpy(state_min).float()
    state_max_tensor = torch.from_numpy(state_max).float()
    state_range = state_max_tensor - state_min_tensor
    state_range = torch.where(state_range > 1e-6, state_range, torch.ones_like(state_range))

    # 创建数据parquet文件
    print("创建数据文件...")

    # 状态归一化 (使用min-max归一化)
    states_tensor = torch.from_numpy(states_array).float()
    normalized_states = (states_tensor - state_min_tensor) / state_range

    # 创建动作序列 (每个样本对应 action_chunk_size 个动作)
    # 对于离散动作，我们使用 one-hot 编码
    action_dim = len(action_to_idx)

    # 动作序列：对于每个时间步，如果有动作则编码为one-hot，否则为全零
    actions_array = np.zeros((num_samples, action_chunk_size, action_dim), dtype=np.float32)

    for i in range(num_samples):
        sample_actions = samples[i].get("actions", [])
        for j in range(min(len(sample_actions), action_chunk_size)):
            action = sample_actions[j]
            if action in action_to_idx:
                actions_array[i, j, action_to_idx[action]] = 1.0

    # 分块写入parquet
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)

        # 将action转换为可序列化的格式 (每个样本的action展平为1D列表)
        actions_list = []
        for i in range(start_idx, end_idx):
            # 将 (action_chunk_size, action_dim) 展平为 (action_chunk_size * action_dim,)
            action_flat = actions_array[i].flatten().tolist()
            actions_list.append(action_flat)

        chunk_data = {
            "observation.image": [f"videos/observation.images.fpv/chunk-{chunk_idx:03d}/frame_{i - start_idx:06d}.jpg"
                                for i in range(start_idx, end_idx)],
            "observation.state": normalized_states[start_idx:end_idx].tolist(),
            "action": actions_list,
        }

        df = pd.DataFrame(chunk_data)
        chunk_file = data_dir / f"chunk-{chunk_idx:03d}"
        chunk_file.mkdir(exist_ok=True)

        for file_idx in range(0, end_idx - start_idx, 100):
            sub_end = min(file_idx + 100, end_idx - start_idx)
            sub_df = df.iloc[file_idx:sub_end]
            sub_file = chunk_file / f"file-{file_idx // 100:03d}.parquet"
            sub_df.to_parquet(sub_file, index=False)

    # 创建元数据
    print("创建元数据...")

    # info.json
    info = {
        "version": "1.0",
        "num_samples": num_samples,
        "action_chunk_size": action_chunk_size,
        "action_dim": action_dim,
        "state_dim": len(state_keys),
        "num_cameras": 1,
        "camera_names": ["fpv"],
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # stats.json
    stats = {
        "state_mean": state_mean.tolist() if hasattr(state_mean, 'tolist') else state_mean,
        "state_std": state_std.tolist() if hasattr(state_std, 'tolist') else state_std,
        "state_min": state_min.tolist() if hasattr(state_min, 'tolist') else state_min,
        "state_max": state_max.tolist() if hasattr(state_max, 'tolist') else state_max,
        "action_counts": action_counts,
    }
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"导出完成! 数据保存在: {output_dir}")
    print(f"  数据文件: {data_dir}")
    print(f"  图像文件: {videos_dir}")
    print(f"  元数据: {meta_dir}")

    return str(output_path)


def create_demo_samples(num_samples: int = 100) -> List[Dict[str, Any]]:
    """创建演示样本用于测试导出功能"""
    samples = []
    for i in range(num_samples):
        # 创建一个简单的测试图像 (1x1 像素的红色 JPEG)
        # 这是一个最小的有效 JPEG 图像
        sample = {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAB//9k=",
            "state": {
                "x": 400 + i * 0.5,
                "y": 300,
                "angle": -np.pi / 2,
                "speed": 1.0,
                "maxSpeed": 5,
                "acceleration": 0.2,
                "rotationSpeed": 0.05,
            },
            "actions": ["forward"] if i % 2 == 0 else [],
        }
        samples.append(sample)
    return samples


if __name__ == "__main__":
    # 测试导出功能
    import state as sim_state

    if sim_state.dataset_samples:
        print(f"使用已采集的 {len(sim_state.dataset_samples)} 个样本导出")
        export_dataset(sim_state.dataset_samples, "dataset")
    else:
        print("没有采集数据，使用演示样本测试导出")
        demo_samples = create_demo_samples(100)
        export_dataset(demo_samples, "dataset")
