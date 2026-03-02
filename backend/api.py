"""
AKA-Sim 后端 - REST API 端点
"""

import logging

from fastapi import APIRouter, HTTPException

import act_model as act_model_module
from config import config
from models import ACTInferenceRequest, DatasetPayload
import state
from data_export import export_dataset

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """根路径"""
    return {
        "name": "AKA-Sim Backend",
        "version": "1.0.0",
        "status": "running",
    }


@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": act_model_module.is_model_loaded(),
    }


@router.post("/api/dataset")
async def save_dataset(payload: DatasetPayload):
    """保存数据集样本"""
    try:
        state.dataset_samples.append({
            "observation": payload.observation,
            "action": payload.action,
        })

        logger.info(f"保存数据集样本，当前共 {len(state.dataset_samples)} 个样本")

        return {
            "success": True,
            "samples_count": len(state.dataset_samples),
        }
    except Exception as e:
        logger.error(f"保存数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dataset")
async def get_dataset():
    """获取数据集"""
    return {
        "samples": state.dataset_samples,
        "count": len(state.dataset_samples),
    }


@router.delete("/api/dataset")
async def clear_dataset():
    """清空数据集"""
    state.dataset_samples = []
    return {
        "success": True,
        "message": "数据集已清空",
    }


@router.post("/api/act/load")
async def load_model(path: str):
    """加载 ACT 模型"""
    try:
        config.MODEL_PATH = path
        act_model_module.load_act_model()
        return {
            "success": True,
            "device": act_model_module.get_model_device(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/act/infer")
async def infer_act(request: ACTInferenceRequest):
    """ACT 模型推理 API"""
    try:
        action = act_model_module.act_inference(request.state)
        return {
            "success": True,
            "action": action,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/dataset/export")
async def export_dataset_api(output_dir: str = "dataset"):
    """导出数据集为ACT训练格式"""
    try:
        if not state.dataset_samples:
            return {
                "success": False,
                "message": "没有采集数据可导出",
            }

        output_path = export_dataset(state.dataset_samples, output_dir)
        return {
            "success": True,
            "output_path": output_path,
            "samples_count": len(state.dataset_samples),
        }
    except Exception as e:
        logger.error(f"导出数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
