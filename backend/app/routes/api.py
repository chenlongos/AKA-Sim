import math
import os
import base64
import sys
import threading
import shutil
import requests
from pathlib import Path
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False
import socket
import struct
import time
import torch

from flask import Blueprint, request, jsonify
from backend.utils.constants import OBS_STATE, OBS_ENV_STATE, ACTION, OBS_IMAGE, OBS_IMAGES, REWARD, DONE, TRUNCATED, OBS_LANGUAGE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ROBOTS, TELEOPERATORS
from backend.sim.model.car import car
from backend.train import train_from_dataset, build_config
from backend.policies.act.modeling_act import ACT
from ..extensions import socketio

api_bp = Blueprint("api", __name__)
ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "output"
DATASET_DIR = OUTPUT_DIR / "datasets"
MODEL_DIR = OUTPUT_DIR / "train"

train_lock = threading.Lock()
train_state = {
    "status": "idle",
    "run_id": None,
    "dataset_path": None,
    "model_id": None,
    "checkpoint_path": None,
    "epoch": 0,
    "num_epochs": 0,
    "avg_loss": None,
    "progress": 0.0,
    "message": None,
    "error": None,
    "started_at": None,
    "ended_at": None,
    "updated_at": None,
}
train_thread = None

infer_lock = threading.Lock()
infer_state = {
    "status": "idle",
    "model_id": None,
    "checkpoint_path": None,
    "device": None,
    "model": None,
    "state_dim": 0,
    "env_state_dim": 0,
    "action_dim": 0,
    "chunk_size": 0,
    # Action chunk caching for temporal ensembling
    "chunk_buffer": [],      # list of past action chunks for ensembling
    "chunk_index": 0,        # current step index within the cached chunk
    "current_chunk": None,   # (chunk_size, action_dim) tensor
}


@api_bp.route('/dataset', methods=['POST'])
def save_dataset():
    payload = request.get_json(silent=True) or {}
    states = payload.get("states") or payload.get(OBS_STATE)
    env_states = payload.get("env_states") or payload.get(OBS_ENV_STATE)
    actions = payload.get("actions") or payload.get(ACTION)
    action_is_pad = payload.get("action_is_pad")
    rewards = payload.get(REWARD)
    dones = payload.get(DONE)
    truncateds = payload.get(TRUNCATED)
    languages = payload.get(OBS_LANGUAGE)
    language_tokens = payload.get(OBS_LANGUAGE_TOKENS)
    language_masks = payload.get(OBS_LANGUAGE_ATTENTION_MASK)
    robots = payload.get(ROBOTS)
    teleoperators = payload.get(TELEOPERATORS)
    images = payload.get(OBS_IMAGES)
    if images is None:
        single_images = payload.get(OBS_IMAGE)
        if single_images is not None:
            images = [[[(img if img is not None else "")] for img in chunk] for chunk in single_images]
    if images is None:
        legacy_images = payload.get("images")
        if legacy_images is not None:
            images = [[[(img if img is not None else "")] for img in chunk] for chunk in legacy_images]
    if not (isinstance(states, list) and isinstance(env_states, list) and isinstance(actions, list) and isinstance(action_is_pad, list)):
        return jsonify({"error": "invalid payload"}), 400
    if not (len(states) == len(env_states) == len(actions) == len(action_is_pad)):
        return jsonify({"error": "length mismatch"}), 400
    if rewards is not None and (not isinstance(rewards, list) or len(rewards) != len(actions)):
        return jsonify({"error": "reward length mismatch"}), 400
    if dones is not None and (not isinstance(dones, list) or len(dones) != len(actions)):
        return jsonify({"error": "done length mismatch"}), 400
    if truncateds is not None and (not isinstance(truncateds, list) or len(truncateds) != len(actions)):
        return jsonify({"error": "truncated length mismatch"}), 400
    if languages is not None and (not isinstance(languages, list) or len(languages) != len(actions)):
        return jsonify({"error": "language length mismatch"}), 400
    if language_tokens is not None and (not isinstance(language_tokens, list) or len(language_tokens) != len(actions)):
        return jsonify({"error": "language tokens length mismatch"}), 400
    if language_masks is not None and (not isinstance(language_masks, list) or len(language_masks) != len(actions)):
        return jsonify({"error": "language masks length mismatch"}), 400
    if images is not None:
        if not isinstance(images, list):
            return jsonify({"error": "invalid images"}), 400
        if len(images) != len(actions):
            return jsonify({"error": "images length mismatch"}), 400
    def _sanitize_float_list(data):
        """Recursively replace None/NaN/Inf with 0.0 to prevent tensor creation errors."""
        if isinstance(data, list):
            return [_sanitize_float_list(item) for item in data]
        if data is None or (isinstance(data, float) and (math.isnan(data) or math.isinf(data))):
            return 0.0
        return data

    dataset = {
        OBS_STATE: torch.tensor(_sanitize_float_list(states), dtype=torch.float32),
        OBS_ENV_STATE: torch.tensor(_sanitize_float_list(env_states), dtype=torch.float32),
        ACTION: torch.tensor(_sanitize_float_list(actions), dtype=torch.float32),
        "action_is_pad": torch.tensor(action_is_pad, dtype=torch.bool),
    }
    if rewards is not None:
        dataset[REWARD] = torch.tensor(rewards, dtype=torch.float32)
    if dones is not None:
        dataset[DONE] = torch.tensor(dones, dtype=torch.bool)
    if truncateds is not None:
        dataset[TRUNCATED] = torch.tensor(truncateds, dtype=torch.bool)
    if languages is not None:
        dataset[OBS_LANGUAGE] = languages
    if language_tokens is not None:
        dataset[OBS_LANGUAGE_TOKENS] = language_tokens
    if language_masks is not None:
        dataset[OBS_LANGUAGE_ATTENTION_MASK] = language_masks
    if isinstance(robots, list):
        dataset[ROBOTS] = robots
    if isinstance(teleoperators, list):
        dataset[TELEOPERATORS] = teleoperators
    meta = payload.get("meta")
    if isinstance(meta, dict):
        dataset["meta"] = meta
    save_dir = os.getenv("ACT_DATASET_DIR", os.path.join("output", "datasets"))
    os.makedirs(save_dir, exist_ok=True)
    timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"
    image_dir = None
    image_count = 0
    image_paths = None
    if images is not None:
        image_dir = os.path.join(save_dir, f"images_{timestamp}")
        os.makedirs(image_dir, exist_ok=True)
        image_paths = []
        for chunk_index, chunk in enumerate(images):
            if not isinstance(chunk, list):
                image_paths.append([])
                continue
            chunk_paths = []
            for step_index, camera_list in enumerate(chunk):
                if not isinstance(camera_list, list):
                    camera_list = [camera_list]
                step_paths = []
                for cam_index, data_url in enumerate(camera_list):
                    saved_name = ""
                    if isinstance(data_url, str) and data_url and "base64," in data_url:
                        _, b64 = data_url.split("base64,", 1)
                        try:
                            raw = base64.b64decode(b64)
                            saved_name = f"chunk_{chunk_index:05d}_step_{step_index:02d}_cam_{cam_index}.png"
                            saved_path = os.path.join(image_dir, saved_name)
                            with open(saved_path, "wb") as f:
                                f.write(raw)
                            image_count += 1
                        except Exception:
                            saved_name = ""
                    if saved_name:
                        step_paths.append(os.path.relpath(saved_path, save_dir))
                    else:
                        step_paths.append("")
                chunk_paths.append(step_paths)
            image_paths.append(chunk_paths)
        dataset[OBS_IMAGES] = image_paths
        if image_count > 0:
            if "meta" not in dataset:
                dataset["meta"] = {}
            dataset["meta"]["image_dir"] = os.path.relpath(image_dir, save_dir)
            dataset["meta"]["image_count"] = image_count
    path = os.path.join(save_dir, f"act_dataset_{timestamp}.pt")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(save_dir, f"act_dataset_{timestamp}_{suffix}.pt")
        suffix += 1
    torch.save(dataset, path)
    return jsonify({"status": "success", "path": path})


def _latest_dataset_path():
    if not DATASET_DIR.exists():
        return None
    candidates = [p for p in DATASET_DIR.glob("*.pt") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _list_datasets():
    if not DATASET_DIR.exists():
        return []
    datasets = []
    for dataset_path in DATASET_DIR.glob("*.pt"):
        if not dataset_path.is_file():
            continue
        stat = dataset_path.stat()
        datasets.append(
            {
                "id": dataset_path.name,
                "path": str(dataset_path),
                "size_bytes": int(stat.st_size),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    datasets.sort(key=lambda d: d["updated_at"], reverse=True)
    return datasets


def _list_models():
    if not MODEL_DIR.exists():
        return []
    models = []
    for model_dir in MODEL_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        checkpoint_path = model_dir / "act_checkpoint.pt"
        if not checkpoint_path.exists():
            continue
        stat = checkpoint_path.stat()
        models.append(
            {
                "id": model_dir.name,
                "path": str(checkpoint_path),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    models.sort(key=lambda m: m["updated_at"], reverse=True)
    return models


def _apply_continuous_action(action_vec, action_dim: int):
    """Apply continuous action [velocity, angular_velocity] to car physics."""
    velocity = float(action_vec[0])
    angular_vel = float(action_vec[1]) if action_dim >= 2 else 0.0

    # Clamp values
    velocity = max(-0.15, min(0.15, velocity))
    angular_vel = max(-0.08, min(0.08, angular_vel))

    car.speed += velocity
    car.angle += angular_vel

    # Friction
    car.speed *= car.friction
    car.x += math.cos(car.angle) * car.speed
    car.y += math.sin(car.angle) * car.speed

    state = car.get_state()
    socketio.emit("car_state", state)


def _run_training(run_id: str, dataset_path: str, options: dict):
    output_dir = MODEL_DIR / f"act_{time.strftime('%Y%m%d_%H%M%S')}"

    def _progress(epoch: int, total: int, avg_loss: float):
        with train_lock:
            train_state.update(
                {
                    "status": "running",
                    "epoch": epoch,
                    "num_epochs": total,
                    "avg_loss": avg_loss,
                    "progress": float(epoch) / float(total) if total else 0.0,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    try:
        result = train_from_dataset(
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            num_epochs=options.get("num_epochs", 50),
            batch_size=options.get("batch_size", 64),
            lr=options.get("lr", 3e-4),
            device=options.get("device"),
            use_vae=options.get("use_vae", False),
            kl_weight=options.get("kl_weight", 1.0),
            grad_clip_norm=options.get("grad_clip_norm", 1.0),
            progress_callback=_progress,
        )
        with train_lock:
            train_state.update(
                {
                    "status": "completed",
                    "model_id": output_dir.name,
                    "checkpoint_path": result["checkpoint_path"],
                    "progress": 1.0,
                    "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    except Exception as exc:
        with train_lock:
            train_state.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )


@api_bp.route("/train/start", methods=["POST"])
def train_start():
    data = request.get_json(silent=True) or {}
    dataset_path_raw = data.get("dataset_path") or _latest_dataset_path()
    if not dataset_path_raw:
        return jsonify({"status": "error", "message": "dataset not found"}), 400
    dataset_path = str(Path(dataset_path_raw).expanduser().resolve())
    if not os.path.exists(dataset_path):
        return jsonify({"status": "error", "message": "dataset not found"}), 400
    with train_lock:
        if train_state["status"] == "running":
            return jsonify({"status": "error", "message": "training already running"}), 400
        run_id = time.strftime("%Y%m%d_%H%M%S")
        train_state.update(
            {
                "status": "starting",
                "run_id": run_id,
                "dataset_path": dataset_path,
                "model_id": None,
                "checkpoint_path": None,
                "epoch": 0,
                "num_epochs": int(data.get("num_epochs", 50)),
                "avg_loss": None,
                "progress": 0.0,
                "message": None,
                "error": None,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": None,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    options = {
        "num_epochs": data.get("num_epochs", 50),
        "batch_size": data.get("batch_size", 64),
        "lr": data.get("lr", 3e-4),
        "device": data.get("device"),
        "use_vae": data.get("use_vae", False),
        "kl_weight": data.get("kl_weight", 1.0),
        "grad_clip_norm": data.get("grad_clip_norm", 1.0),
    }
    global train_thread
    train_thread = threading.Thread(
        target=_run_training,
        args=(run_id, dataset_path, options),
        daemon=True,
    )
    train_thread.start()
    return jsonify({"status": "started", "run_id": run_id, "dataset_path": dataset_path})


@api_bp.route("/train/status")
def train_status():
    with train_lock:
        return jsonify(dict(train_state))


@api_bp.route("/datasets")
def datasets():
    return jsonify({"datasets": _list_datasets()})


@api_bp.route("/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id: str):
    if "/" in dataset_id or "\\" in dataset_id:
        return jsonify({"status": "error", "message": "invalid dataset id"}), 400
    dataset_path = (DATASET_DIR / dataset_id).resolve()
    dataset_root = DATASET_DIR.resolve()
    try:
        dataset_path.relative_to(dataset_root)
    except ValueError:
        return jsonify({"status": "error", "message": "invalid dataset id"}), 400
    if dataset_path.suffix != ".pt":
        return jsonify({"status": "error", "message": "invalid dataset file"}), 400
    if not dataset_path.exists():
        return jsonify({"status": "error", "message": "dataset not found"}), 404
    with train_lock:
        if train_state["status"] in {"starting", "running"} and train_state.get("dataset_path") == str(dataset_path):
            return jsonify({"status": "error", "message": "dataset is in active training"}), 400
    image_dir_from_meta = None
    try:
        dataset_obj = torch.load(str(dataset_path), map_location="cpu")
        meta = dataset_obj.get("meta")
        if isinstance(meta, dict):
            image_dir = meta.get("image_dir")
            if isinstance(image_dir, str) and image_dir:
                image_dir_from_meta = (dataset_root / image_dir).resolve()
    except Exception:
        image_dir_from_meta = None
    os.remove(dataset_path)
    removed_image_dir = False
    if image_dir_from_meta is not None:
        try:
            image_dir_from_meta.relative_to(dataset_root)
            if image_dir_from_meta.exists() and image_dir_from_meta.is_dir():
                shutil.rmtree(image_dir_from_meta)
                removed_image_dir = True
        except Exception:
            pass
    if not removed_image_dir:
        stem = dataset_path.stem
        if stem.startswith("act_dataset_"):
            ts = stem.replace("act_dataset_", "", 1)
            fallback_image_dir = dataset_root / f"images_{ts}"
            if fallback_image_dir.exists() and fallback_image_dir.is_dir():
                shutil.rmtree(fallback_image_dir)
                removed_image_dir = True
    return jsonify(
        {
            "status": "deleted",
            "dataset_id": dataset_id,
            "removed_image_dir": removed_image_dir,
        }
    )


@api_bp.route("/models")
def models():
    return jsonify({"models": _list_models()})


@api_bp.route("/models/<model_id>", methods=["DELETE"])
def delete_model(model_id: str):
    if "/" in model_id or "\\" in model_id:
        return jsonify({"status": "error", "message": "invalid model id"}), 400
    model_dir = (MODEL_DIR / model_id).resolve()
    model_root = MODEL_DIR.resolve()
    try:
        model_dir.relative_to(model_root)
    except ValueError:
        return jsonify({"status": "error", "message": "invalid model id"}), 400
    if not model_dir.exists() or not model_dir.is_dir():
        return jsonify({"status": "error", "message": "model not found"}), 404
    with infer_lock:
        if infer_state["status"] == "running" and infer_state.get("model_id") == model_id:
            infer_state.update(
                {
                    "status": "idle",
                    "model_id": None,
                    "checkpoint_path": None,
                    "device": None,
                    "model": None,
                    "state_dim": 0,
                    "env_state_dim": 0,
                    "action_dim": 0,
                    "chunk_size": 0,
                }
            )
    shutil.rmtree(model_dir)
    return jsonify({"status": "deleted", "model_id": model_id})


@api_bp.route("/infer/start", methods=["POST"])
def infer_start():
    data = request.get_json(silent=True) or {}
    model_id = data.get("model_id")
    model_path = data.get("model_path")
    if model_path is None and model_id:
        model_path = str(MODEL_DIR / model_id / "act_checkpoint.pt")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "model not found"}), 400
    device = data.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dim = int(checkpoint["state_dim"])
        env_state_dim = int(checkpoint["env_state_dim"])
        action_dim = int(checkpoint["action_dim"])
        chunk_size = int(checkpoint["chunk_size"])
        use_vae = bool(checkpoint.get("use_vae", False))
        config = build_config(state_dim, env_state_dim, action_dim, chunk_size, use_vae)
        model = ACT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"failed to load model: {exc}"}), 400
    with infer_lock:
        infer_state.update(
            {
                "status": "running",
                "model_id": model_id or Path(model_path).parent.name,
                "checkpoint_path": model_path,
                "device": device,
                "model": model,
                "state_dim": state_dim,
                "env_state_dim": env_state_dim,
                "action_dim": action_dim,
                "chunk_size": chunk_size,
            }
        )
    return jsonify({"status": "started", "model_id": infer_state["model_id"]})


@api_bp.route("/infer/step", methods=["POST"])
def infer_step():
    with infer_lock:
        if infer_state["status"] != "running":
            return jsonify({"status": "error", "message": "inference not running"}), 400
        model = infer_state["model"]
        device = infer_state["device"]
        state_dim = infer_state["state_dim"]
        env_state_dim = infer_state["env_state_dim"]
        action_dim = infer_state["action_dim"]
        chunk_size = infer_state["chunk_size"]

    def _to_fixed_vec(raw, dim: int):
        vec = [0.0] * dim
        if not isinstance(raw, list):
            return vec
        for i in range(min(dim, len(raw))):
            try:
                vec[i] = float(raw[i])
            except (TypeError, ValueError):
                vec[i] = 0.0
        return vec

    data = request.get_json(silent=True) or {}
    state_input = data.get("state")
    env_input = data.get("env_state")
    force_replan = data.get("force_replan", False)

    if state_input is None or env_input is None:
        state = car.get_state()
        state_vec = [0.0] * state_dim
        state_vec[0:3] = [float(state["x"]), float(state["y"]), float(state["angle"])]
        if state_dim > 3:
            state_vec[3] = float(car.speed)
        env_vec = [0.0] * env_state_dim
    else:
        state_vec = _to_fixed_vec(state_input, state_dim)
        env_vec = _to_fixed_vec(env_input, env_state_dim)

    with infer_lock:
        chunk_idx = infer_state["chunk_index"]
        current_chunk = infer_state["current_chunk"]
        chunk_buffer = infer_state["chunk_buffer"]

        # Run model forward pass to get new action chunk
        state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=device)
        env_tensor = torch.tensor([env_vec], dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _ = model({OBS_STATE: state_tensor, OBS_ENV_STATE: env_tensor})
        # actions shape: (1, chunk_size, action_dim)
        new_chunk = actions[0].detach().cpu()  # (chunk_size, action_dim)

        # Temporal ensembling: average with past chunks using exponential decay
        chunk_buffer.append(new_chunk)
        if len(chunk_buffer) > chunk_size:
            chunk_buffer.pop(0)

        if len(chunk_buffer) >= 2:
            ensembled = torch.zeros_like(new_chunk)
            total_weight = 0.0
            for i, past_chunk in enumerate(chunk_buffer):
                weight = 0.5 ** (len(chunk_buffer) - 1 - i)  # exponential decay
                ensembled += past_chunk * weight
                total_weight += weight
            ensembled /= total_weight
        else:
            ensembled = new_chunk

        # Cache the ensembled chunk and use step-by-step
        infer_state["current_chunk"] = ensembled
        infer_state["chunk_index"] = 0
        action_vec = ensembled[0].numpy().tolist()

    _apply_continuous_action(action_vec, action_dim)
    return jsonify({
        "status": "ok",
        "action": action_vec,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
    })


@api_bp.route("/infer/stop", methods=["POST"])
def infer_stop():
    with infer_lock:
        infer_state.update(
            {
                "status": "idle",
                "model_id": None,
                "checkpoint_path": None,
                "device": None,
                "model": None,
                "state_dim": 0,
                "env_state_dim": 0,
                "action_dim": 0,
                "chunk_size": 0,
                "chunk_buffer": [],
                "chunk_index": 0,
                "current_chunk": None,
            }
        )
    return jsonify({"status": "stopped"})


@api_bp.route("/control")
def control():
    """向指定IP发送控制指令"""
    action = request.args.get("action", "stop")
    speed = request.args.get("speed", "50")
    time_sec = request.args.get("time", "0")
    target_ip = request.args.get("target_ip")

    if not target_ip:
        # 如果没有指定IP，返回本机IP
        return jsonify({"error": "missing target_ip"}), 400

    try:
        target_url = f"http://{target_ip}/api/control"
        params = {
            "action": action,
            "speed": speed,
            "time": time_sec,
        }
        resp = requests.get(target_url, params=params, timeout=5)
        return jsonify({"status": "ok", "response": resp.text})
    except requests.exceptions.Timeout:
        return jsonify({"error": f"timeout: cannot reach {target_ip}"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"connection failed: cannot reach {target_ip}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/ip")
def get_ip():
    """获取本机IP地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 连接到一个外部IP（不实际发送数据）
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()
    return jsonify({"ip": local_ip})