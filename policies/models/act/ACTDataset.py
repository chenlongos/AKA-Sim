from typing import Dict, Tuple

import torch


class ACTDataset(torch.utils.data.Dataset):
    """
    ACT 数据集 - 用于行为克隆训练
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        action_chunk_size: int = 16,
        normalize_images: bool = True,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            data: 包含 'observation.image', 'observation.state', 'action' 的字典
            action_chunk_size: 动作分块大小
        """
        self.data = data
        self.action_chunk_size = action_chunk_size

        # 归一化参数
        self.normalize_images = normalize_images
        self.image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std).view(1, 3, 1, 1)

        # 计算数据集大小
        # 动作序列长度
        self.num_samples = data["action"].shape[0] - action_chunk_size + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            sample: 包含 'observation' 和 'action' 的字典
        """
        # 获取动作块对应的观察
        # 假设观察是每个时间步都有，这里取最后一个时间步的观察作为当前观察
        # 和前 action_chunk_size - 1 个历史观察

        # 获取当前时间步的观察
        current_idx = idx + self.action_chunk_size - 1

        # 支持两种数据格式
        if "observation.image" in self.data:
            images = self.data["observation.image"][current_idx]
            state = self.data["observation.state"][current_idx]
        else:
            images = self.data["observation"]["image"][current_idx]
            state = self.data["observation"]["state"][current_idx]

        action = self.data["action"][idx:idx + self.action_chunk_size]

        # 归一化图像
        if self.normalize_images:
            images = (images - self.image_mean.to(images.device)) / self.image_std.to(images.device)

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
