from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cameras.opencv.camera_opencv import OpenCVCamera


ACTION_KEYS = {
    "w": "up",
    "s": "down",
    "a": "left",
    "d": "right",
    " ": "stop",
    "x": "stop",
}


@dataclass
class ClientConfig:
    backend_url: str
    robot_base_url: str | None
    telemetry_url: str | None
    state_dim: int
    env_dim: int
    camera_device: int
    camera_width: int
    camera_height: int
    speed: int
    action_time: float
    timeout: float
    dry_run: bool = False


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected JSON payload from {url}")
    return payload


def get_observation(
    *,
    telemetry_url: str | None,
    state_dim: int,
    env_dim: int,
    image_data_url: str | None = None,
) -> dict[str, Any]:
    state = [0.0] * state_dim
    environment_state = [0.0] * env_dim

    if telemetry_url:
        payload = fetch_json(telemetry_url, timeout=5.0)
        obs = payload.get("observation") if isinstance(payload.get("observation"), dict) else payload
        raw_state = obs.get("state") if isinstance(obs, dict) else None
        raw_env = None
        if isinstance(obs, dict):
            raw_env = obs.get("environment_state")
            if raw_env is None:
                raw_env = obs.get("env_state")
        state = fixed_vector(raw_state, state_dim)
        environment_state = fixed_vector(raw_env, env_dim)

    observation: dict[str, Any] = {
        "state": state,
        "environment_state": environment_state,
    }
    if image_data_url:
        observation["cameras"] = [
            {
                "camera_id": "front",
                "data_url": image_data_url,
                "timestamp_ms": int(time.time() * 1000),
            }
        ]
    return observation


def fixed_vector(raw: Any, dim: int) -> list[float]:
    vector = [0.0] * dim
    if not isinstance(raw, list):
        return vector
    for index, value in enumerate(raw[:dim]):
        try:
            vector[index] = float(value)
        except (TypeError, ValueError):
            vector[index] = 0.0
    return vector


def send_robot_action(
    *,
    robot_base_url: str | None,
    action: str,
    speed: int,
    action_time: float,
    timeout: float,
    dry_run: bool = False,
) -> None:
    if dry_run or not robot_base_url:
        return
    response = requests.get(
        f"{normalize_url(robot_base_url)}/api/control",
        params={"action": action, "speed": speed, "time": action_time},
        timeout=timeout,
    )
    response.raise_for_status()


def backend_post_json(backend_url: str, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    response = requests.post(
        f"{normalize_url(backend_url)}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"unexpected JSON payload from {path}")
    return data


def open_camera(config: ClientConfig) -> OpenCVCamera:
    return OpenCVCamera(
        device_index=config.camera_device,
        width=config.camera_width,
        height=config.camera_height,
    )


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))
