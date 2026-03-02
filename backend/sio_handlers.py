"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

import asyncio
import base64
import io
import logging

from socketio import AsyncNamespace

import state

logger = logging.getLogger(__name__)

# 当前按下的按键集合
current_actions = set()

# 数据采集状态
is_collecting = False


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
        logger.info(f"收到动作: {actions}, 类型: {type(actions)}")
        current_actions = set(actions)
        logger.info(f"当前动作集合: {current_actions}")

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

    async def on_set_collection(self, sid: str, enabled: bool):
        """设置数据采集状态"""
        global is_collecting
        is_collecting = enabled
        logger.info(f"数据采集{'开启' if enabled else '关闭'}")

        if not enabled and state.dataset_samples:
            # 停止采集时自动导出
            logger.info("停止采集，自动导出数据...")
            try:
                from data_export import export_dataset
                output_path = export_dataset(state.dataset_samples)
                logger.info(f"数据已导出到: {output_path}")
                await self.emit("collection_count", {
                    "count": len(state.dataset_samples),
                    "exported": True,
                    "output_path": output_path
                })
            except Exception as e:
                logger.error(f"自动导出失败: {e}")
                await self.emit("collection_count", {
                    "count": len(state.dataset_samples),
                    "exported": False,
                    "error": str(e)
                })
        else:
            await self.emit("collection_count", {"count": len(state.dataset_samples)})

    async def on_collect_data(self, sid: str, payload: dict):
        """接收前端发送的图像数据进行保存"""
        global is_collecting

        if not is_collecting:
            return

        try:
            # 解析图像数据 (base64)
            image_data = payload.get("image", "")
            actions = payload.get("actions", [])

            # 获取当前车辆状态
            car_state = state.car_state.copy()

            # 保存样本
            sample = {
                "image": image_data,  # base64编码的JPEG图像
                "state": car_state,
                "actions": actions,
            }
            state.dataset_samples.append(sample)

            # 定期广播计数 (每10个样本)
            if len(state.dataset_samples) % 10 == 0:
                await self.emit("collection_count", {"count": len(state.dataset_samples)})

        except Exception as e:
            logger.error(f"数据采集失败: {e}")


# 游戏循环任务
async def game_loop_task(sio_server):
    """游戏循环 - 处理物理更新和状态广播"""
    global current_actions

    while True:
        try:
            # 根据当前按键更新状态
            if current_actions:
                logger.info(f"游戏循环处理动作: {current_actions}")
                for action in current_actions:
                    state.update_car_state(action)
                logger.info(f"更新后状态: x={state.car_state['x']}, y={state.car_state['y']}, speed={state.car_state['speed']}")
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
