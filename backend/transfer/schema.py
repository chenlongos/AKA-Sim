from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransferMode(str, Enum):
    REAL_TO_REAL = "real_to_real"
    SIM_TO_REAL = "sim_to_real"


class SourceDomain(str, Enum):
    REAL = "real"
    SIM = "sim"


@dataclass
class CameraFrame:
    camera_id: str = "front"
    data_url: str = ""
    width: int | None = None
    height: int | None = None
    encoding: str | None = None
    timestamp_ms: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraFrame":
        return cls(
            camera_id=str(payload.get("camera_id") or "front"),
            data_url=str(payload.get("data_url") or ""),
            width=_maybe_int(payload.get("width")),
            height=_maybe_int(payload.get("height")),
            encoding=_maybe_str(payload.get("encoding")),
            timestamp_ms=_maybe_int(payload.get("timestamp_ms")),
            extras=dict(payload.get("extras") or {}),
        )


@dataclass
class ControlAction:
    command: str | None = None
    vector: list[float] | None = None
    throttle: float | None = None
    steering: float | None = None
    speed: float | None = None
    duration_ms: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControlAction":
        vector = payload.get("vector")
        parsed_vector: list[float] | None = None
        if isinstance(vector, list):
            parsed_vector = [_to_float(v) for v in vector]
        return cls(
            command=_maybe_str(payload.get("command")),
            vector=parsed_vector,
            throttle=_maybe_float(payload.get("throttle")),
            steering=_maybe_float(payload.get("steering")),
            speed=_maybe_float(payload.get("speed")),
            duration_ms=_maybe_int(payload.get("duration_ms")),
            extras=dict(payload.get("extras") or {}),
        )


@dataclass
class ObservationFrame:
    state: list[float]
    environment_state: list[float]
    cameras: list[CameraFrame] = field(default_factory=list)
    timestamp_ms: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationFrame":
        cameras_payload = payload.get("cameras") or payload.get("images") or []
        cameras = [
            CameraFrame.from_dict(item)
            for item in cameras_payload
            if isinstance(item, dict)
        ]
        return cls(
            state=[_to_float(v) for v in list(payload.get("state") or [])],
            environment_state=[_to_float(v) for v in list(payload.get("environment_state") or payload.get("env_state") or [])],
            cameras=cameras,
            timestamp_ms=_maybe_int(payload.get("timestamp_ms")),
            extras=dict(payload.get("extras") or {}),
        )


@dataclass
class TransferStep:
    observation: ObservationFrame
    action: ControlAction
    reward: float | None = None
    done: bool = False
    truncated: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransferStep":
        observation_payload = payload.get("observation") or {}
        action_payload = payload.get("action") or {}
        return cls(
            observation=ObservationFrame.from_dict(observation_payload),
            action=ControlAction.from_dict(action_payload),
            reward=_maybe_float(payload.get("reward")),
            done=bool(payload.get("done", False)),
            truncated=bool(payload.get("truncated", False)),
            extras=dict(payload.get("extras") or {}),
        )


@dataclass
class TransferEpisode:
    episode_id: str
    source_domain: SourceDomain
    transfer_mode: TransferMode
    robot_id: str | None = None
    teleoperator_id: str | None = None
    steps: list[TransferStep] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: int = 0) -> "TransferEpisode":
        steps_payload = payload.get("steps") or []
        steps = [
            TransferStep.from_dict(item)
            for item in steps_payload
            if isinstance(item, dict)
        ]
        return cls(
            episode_id=str(payload.get("episode_id") or f"episode_{index:04d}"),
            source_domain=_parse_enum(SourceDomain, payload.get("source_domain"), SourceDomain.REAL),
            transfer_mode=_parse_enum(TransferMode, payload.get("transfer_mode"), TransferMode.REAL_TO_REAL),
            robot_id=_maybe_str(payload.get("robot_id")),
            teleoperator_id=_maybe_str(payload.get("teleoperator_id")),
            steps=steps,
            extras=dict(payload.get("extras") or {}),
        )


def _parse_enum(enum_type: type[Enum], value: Any, default: Enum) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        for item in enum_type:
            if item.value == value:
                return item
    return default


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
