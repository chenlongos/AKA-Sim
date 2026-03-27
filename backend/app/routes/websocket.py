import math
import os
import torch

from backend.sim.model.car import car
from ..extensions import socketio
from flask_socketio import emit
from backend.policies.act.configuration_act import ACTConfig
from backend.policies.act.modeling_act import ACT
from backend.utils.constants import OBS_STATE, OBS_ENV_STATE, ACTION
from backend.configs.types import PolicyFeature, FeatureType

_act_model = None
_act_device = None


@socketio.on('action')
def handle_action(action):
    if action == 'up':
        if car.speed < car.maxSpeed:
            car.speed += car.acceleration
    if action == 'down':
        if car.speed > -car.maxSpeed / 2:
            car.speed -= car.acceleration
    if action == 'left':
        car.angle -= car.rotationSpeed
    if action == 'right':
        car.angle += car.rotationSpeed

    car.speed *= car.friction
    car.x += math.cos(car.angle) * car.speed
    car.y += math.sin(car.angle) * car.speed

    if action == 'stop':
        car.x -= math.cos(car.angle) * car.speed * 2
        car.y -= math.sin(car.angle) * car.speed * 2
        car.speed = 0
    state = car.get_state()
    emit('car_state', state)


@socketio.on('get_car_state')
def get_car_state():
    status = car.get_state()
    emit('car_state', status)


@socketio.on('reset_car_state')
def get_car_state():
    car.reset()
    status = car.get_state()
    emit('car_state', status)


def _load_act_model():
    global _act_model, _act_device
    if _act_model is not None:
        return
    _act_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.getenv(
        "ACT_CHECKPOINT_PATH",
        os.path.join("output", "train", "act_demo", "act_checkpoint.pt"),
    )
    # Try to load dimensions from checkpoint first
    try:
        checkpoint = torch.load(checkpoint_path, map_location=_act_device)
        state_dim = int(checkpoint.get("state_dim", 14))
        env_state_dim = int(checkpoint.get("env_state_dim", 10))
        action_dim = int(checkpoint.get("action_dim", 2))
        chunk_size = int(checkpoint.get("chunk_size", 10))
        use_vae = bool(checkpoint.get("use_vae", False))
    except Exception:
        # Fallback to env vars
        state_dim = int(os.getenv("ACT_STATE_DIM", "14"))
        env_state_dim = int(os.getenv("ACT_ENV_STATE_DIM", "10"))
        action_dim = int(os.getenv("ACT_ACTION_DIM", "2"))
        chunk_size = int(os.getenv("ACT_CHUNK_SIZE", "10"))
        use_vae = False
    config = ACTConfig(
        chunk_size=chunk_size,
        use_vae=use_vae,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(env_state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
    )
    model = ACT(config).to(_act_device)
    state = torch.load(checkpoint_path, map_location=_act_device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    _act_model = model


def _to_tensor(data):
    tensor = torch.as_tensor(data, dtype=torch.float32, device=_act_device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


@socketio.on('act_infer')
def act_infer(payload):
    try:
        _load_act_model()
        if not isinstance(payload, dict):
            emit('act_action', {"error": "invalid payload"})
            return
        obs = payload.get("observation")
        if not isinstance(obs, dict):
            emit('act_action', {"error": "missing observation"})
            return
        if "state" not in obs or "environment_state" not in obs:
            emit('act_action', {"error": "missing state or environment_state"})
            return
        batch = {
            OBS_STATE: _to_tensor(obs["state"]),
            OBS_ENV_STATE: _to_tensor(obs["environment_state"]),
        }
        with torch.no_grad():
            actions, _ = _act_model(batch)
        emit('act_action', {"action": actions.detach().cpu().tolist()})
    except Exception as exc:
        emit('act_action', {"error": str(exc)})
