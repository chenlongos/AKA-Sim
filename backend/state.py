"""
AKA-Sim 后端 - 全局状态
"""

from typing import TYPE_CHECKING, Optional

from config import config

if TYPE_CHECKING:
    from policies.models.act.act_model_pytorch import ACTModel

# 车辆状态
car_state = {
    "x": 400,
    "y": 300,
    "angle": -3.14159 / 2,  # -90度
    "speed": 0,
    "maxSpeed": 5,
    "acceleration": 0.2,
    "friction": 0.98,
    "rotationSpeed": 0.05,
}

# ACT 模型 - 实际存储在 act_model 模块中，这里仅作为便捷引用
# act_model: Optional["ACTModel"] = None
# model_device = "cuda"

# 数据集存储
dataset_samples = []


def reset_car_state():
    """重置车辆状态"""
    car_state["x"] = 400
    car_state["y"] = 300
    car_state["angle"] = -3.14159 / 2
    car_state["speed"] = 0


def update_car_state(action: str):
    """更新车辆状态"""
    if action == "forward":
        car_state["speed"] = min(
            car_state["speed"] + car_state["acceleration"],
            car_state["maxSpeed"]
        )
    elif action == "backward":
        car_state["speed"] = max(
            car_state["speed"] - car_state["acceleration"],
            -car_state["maxSpeed"] / 2
        )
    elif action == "left":
        car_state["angle"] -= car_state["rotationSpeed"]
    elif action == "right":
        car_state["angle"] += car_state["rotationSpeed"]
    elif action == "stop":
        car_state["speed"] = 0

    # 更新位置
    import math
    car_state["x"] += math.cos(car_state["angle"]) * car_state["speed"]
    car_state["y"] += math.sin(car_state["angle"]) * car_state["speed"]

    # 边界检测
    car_state["x"] = max(0, min(config.MAP_WIDTH, car_state["x"]))
    car_state["y"] = max(0, min(config.MAP_HEIGHT, car_state["y"]))


def apply_friction():
    """应用摩擦力减速"""
    if car_state["speed"] > 0:
        car_state["speed"] = max(0, car_state["speed"] * car_state["friction"])
    elif car_state["speed"] < 0:
        car_state["speed"] = min(0, car_state["speed"] * car_state["friction"])

    # 摩擦力时也要更新位置（小车滑行）
    import math
    car_state["x"] += math.cos(car_state["angle"]) * car_state["speed"]
    car_state["y"] += math.sin(car_state["angle"]) * car_state["speed"]

    # 边界检测
    car_state["x"] = max(0, min(config.MAP_WIDTH, car_state["x"]))
    car_state["y"] = max(0, min(config.MAP_HEIGHT, car_state["y"]))
