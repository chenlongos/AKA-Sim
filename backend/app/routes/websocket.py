import math
import os
import torch

from backend.sim.model.car import car
from ..extensions import socketio
from flask_socketio import emit
from backend.policies.act.modeling_act import ACT
from backend.utils.constants import OBS_STATE, OBS_ENV_STATE, ACTION, OBS_IMAGES, OBS_IMAGE
from backend.utils.image_utils import build_infer_image_tensors
from backend.train import build_config

_act_model = None
_act_device = None
_act_num_cameras = 0
_act_image_size = (224, 224)


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
    global _act_model, _act_device, _act_num_cameras, _act_image_size
    if _act_model is not None:
        return
    _act_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.getenv(
        "ACT_CHECKPOINT_PATH",
        os.path.join("output", "train", "act_demo", "act_checkpoint.pt"),
    )
    checkpoint = torch.load(checkpoint_path, map_location=_act_device)
    state_dim = int(checkpoint.get("state_dim", os.getenv("ACT_STATE_DIM", "14")))
    env_state_dim = int(checkpoint.get("env_state_dim", os.getenv("ACT_ENV_STATE_DIM", "6")))
    action_dim = int(checkpoint.get("action_dim", os.getenv("ACT_ACTION_DIM", "7")))
    chunk_size = int(checkpoint.get("chunk_size", os.getenv("ACT_CHUNK_SIZE", "16")))
    use_vae = bool(checkpoint.get("use_vae", False))
    _act_num_cameras = int(checkpoint.get("num_cameras", 0)) if checkpoint.get("has_images", False) else 0
    raw_image_size = checkpoint.get("image_size") or [224, 224]
    _act_image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
    config = build_config(
        state_dim,
        env_state_dim,
        action_dim,
        chunk_size,
        use_vae,
        num_cameras=_act_num_cameras,
        image_size=_act_image_size,
    )
    model = ACT(config).to(_act_device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
        images = obs.get("images")
        if images is None:
            images = obs.get(OBS_IMAGES)
        if images is None:
            images = obs.get(OBS_IMAGE)
        if _act_num_cameras > 0:
            batch[OBS_IMAGES] = [
                image_tensor.to(_act_device)
                for image_tensor in build_infer_image_tensors(
                    images,
                    num_cameras=_act_num_cameras,
                    image_size=_act_image_size,
                )
            ]
        with torch.no_grad():
            actions, _ = _act_model(batch)
        emit('act_action', {"action": actions.detach().cpu().tolist()})
    except Exception as exc:
        emit('act_action', {"error": str(exc)})
