"""
AKA-Sim 后端 - ACT 模型模块
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from config import config

if TYPE_CHECKING:
    from policies.models.act.act_model_pytorch import ACTModel, ACTConfig

logger = logging.getLogger(__name__)

# 模块级变量
_act_model: Optional["ACTModel"] = None
_model_device = "cpu"


def create_act_config() -> "ACTConfig":
    """创建 ACT 模型配置"""
    from policies.models.act.act_model_pytorch import ACTConfig as PyACTConfig

    return PyACTConfig(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        action_chunk_size=config.ACTION_CHUNK_SIZE,
        hidden_dim=config.HIDDEN_DIM,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_attention_heads=8,
        use_cvae=True,
        use_temporal_ensembling=True,
        use_spatial_softmax=True,
        latent_dim=32,
    )


def load_act_model() -> "ACTModel":
    """加载 ACT 模型"""
    global _act_model

    from policies.models.act.act_model_pytorch import ACTModel as PyACTModel

    logger.info("加载 ACT 模型...")

    model_config = create_act_config()
    _act_model = PyACTModel(model_config)

    # 如果有预训练模型，加载它
    if config.MODEL_PATH and os.path.exists(config.MODEL_PATH):
        logger.info(f"从 {config.MODEL_PATH} 加载模型权重")
        state_dict = torch.load(config.MODEL_PATH, map_location=_model_device)
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
        return [[0.0] * config.ACTION_DIM for _ in range(config.ACTION_CHUNK_SIZE)]

    with torch.no_grad():
        # 准备输入
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(_model_device)

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
