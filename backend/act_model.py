"""
AKA-Sim 后端 - ACT 模型模块
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from config import config

if TYPE_CHECKING:
    from policies.models.act.act_model_pytorch import ACTModel, ACTConfig

logger = logging.getLogger(__name__)

# 模块级变量
_act_model: Optional["ACTModel"] = None
_model_device = "cpu"


def create_act_config(
    state_dim: int = 7,
    action_dim: int = 5,
    action_chunk_size: int = 16,
    hidden_dim: int = 256,
) -> "ACTConfig":
    """创建 ACT 模型配置"""
    from policies.models.act.act_model_pytorch import ACTConfig as PyACTConfig

    return PyACTConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_attention_heads=4,
        use_cvae=False,
        use_temporal_ensembling=False,
        use_spatial_softmax=True,
        latent_dim=16,
    )


def load_act_model(model_path: str = None) -> "ACTModel":
    """加载 ACT 模型"""
    global _act_model

    from policies.models.act.act_model_pytorch import ACTModel as PyACTModel

    logger.info("加载 ACT 模型...")

    # 尝试从checkpoints目录加载训练好的模型
    if model_path is None:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "checkpoints" / "final_model.pt"

    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}")
        _act_model = None
        return None

    logger.info(f"从 {model_path} 加载模型权重")

    # 加载state_dict获取维度信息
    state_dict = torch.load(model_path, map_location=_model_device, weights_only=True)

    # 推断维度 - 使用固定的训练配置
    model_config = create_act_config(
        state_dim=7,
        action_dim=5,
        action_chunk_size=16,
        hidden_dim=256,
    )
    _act_model = PyACTModel(model_config)
    _act_model.load_state_dict(state_dict)

    _act_model = _act_model.to(_model_device)
    _act_model.eval()

    logger.info(f"ACT 模型加载完成，使用设备: {_model_device}")
    return _act_model


def act_inference(state: list) -> list:
    """
    ACT 模型推理

    Args:
        state: 状态向量 [state_dim]

    Returns:
        预测的动作序列 [action_chunk_size, action_dim]
    """
    if _act_model is None:
        logger.warning("ACT 模型未加载，返回随机动作")
        action_dim = 5  # 使用训练时的维度
        action_chunk_size = 16
        return [[0.0] * action_dim for _ in range(action_chunk_size)]

    with torch.no_grad():
        # 准备输入
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(_model_device)

        # 获取模型配置的维度
        action_dim = _act_model.config.action_dim
        action_chunk_size = _act_model.config.action_chunk_size

        # 创建虚拟图像输入 (实际使用时应该传入真实图像)
        image_tensor = torch.randn(1, 1, 3, 224, 224).to(_model_device)

        # 推理
        action = _act_model.get_action(
            image_tensor,
            state_tensor,
            use_temporal_ensembling=False,
        )

        # 转换为 Python 列表
        action_list = action.cpu().numpy()[0].tolist()

    return action_list


def is_model_loaded() -> bool:
    """检查模型是否已加载"""
    return _act_model is not None


def get_model_device() -> str:
    """获取模型设备"""
    return _model_device
