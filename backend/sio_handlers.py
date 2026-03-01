"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

import asyncio
import logging

from socketio import AsyncNamespace

import state

logger = logging.getLogger(__name__)

# 当前按下的按键集合
current_actions = set()


class SimNamespace(AsyncNamespace):
    """模拟器 Socket.IO 命名空间"""

    async def on_connect(self, sid: str, environ: dict):
        """客户端连接"""
        logger.info(f"客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})
        # 发送当前车辆状态
        await self.emit("car_state_update", state.car_state)

    async def on_disconnect(self, sid: str):
        """客户端断开"""
        logger.info(f"客户端断开: {sid}")

    async def on_action(self, sid: str, actions: list):
        """处理动作命令 - 接收前端发送的当前按键列表"""
        global current_actions
        current_actions = set(actions)

        # 如果没有按键，立刻减速到0
        if not actions:
            state.car_state["speed"] = 0

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
            from act_model import act_inference
            action = act_inference(inference_state)

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


# 游戏循环任务
async def game_loop_task(sio_server):
    """游戏循环 - 处理物理更新和状态广播"""
    global current_actions

    while True:
        try:
            # 根据当前按键更新状态
            if current_actions:
                for action in current_actions:
                    state.update_car_state(action)
            else:
                # 没有按键时应用摩擦力减速
                state.apply_friction()

            # 广播状态
            await sio_server.emit("car_state_update", state.car_state)
        except Exception as e:
            logger.error(f"游戏循环错误: {e}")

        # 30 FPS
        await asyncio.sleep(1 / 30)


def start_game_loop(sio_server):
    """启动游戏循环"""
    asyncio.create_task(game_loop_task(sio_server))
