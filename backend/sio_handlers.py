"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

import logging

from socketio import AsyncNamespace

import state
import act_model as act_model_module

logger = logging.getLogger(__name__)


class SimNamespace(AsyncNamespace):
    """模拟器 Socket.IO 命名空间"""

    async def on_connect(self, sid: str, environ: dict):
        """客户端连接"""
        logger.info(f"客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})

    async def on_disconnect(self, sid: str):
        """客户端断开"""
        logger.info(f"客户端断开: {sid}")

    async def on_action(self, sid: str, action: str):
        """处理动作命令"""
        logger.info(f"收到动作命令: {action}")

        # 更新车辆状态
        state.update_car_state(action)

        # 广播状态
        await self.emit("car_state_update", state.car_state)

    async def on_reset_car_state(self, sid: str):
        """重置车辆状态"""
        logger.info("重置车辆状态")
        state.reset_car_state()
        await self.emit("car_state_update", state.car_state)

    async def on_get_car_state(self, sid: str):
        """获取车辆状态"""
        await self.emit("car_state_update", state.car_state)

    async def on_act_infer(self, sid: str, payload: dict):
        """ACT 模型推理"""
        logger.info(f"收到 ACT 推理请求: {payload}")

        try:
            import config as cfg
            inference_state = payload.get("state", [0.0] * cfg.config.STATE_DIM)
            action = act_model_module.act_inference(inference_state)

            await self.emit("act_infer_result", {
                "success": True,
                "action": action,
            })
        except Exception as e:
            logger.error(f"ACT 推理失败: {e}")
            await self.emit("act_infer_result", {
                "success": False,
                "error": str(e),
            })
