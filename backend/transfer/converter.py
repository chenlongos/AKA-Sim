from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from backend.transfer.schema import TransferEpisode, TransferStep
from backend.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, ROBOTS, TELEOPERATORS


DEFAULT_ACTION_LABELS = ["up", "down", "left", "right", "stop"]


def episodes_from_payload(payload: Any) -> list[TransferEpisode]:
    if not isinstance(payload, list):
        raise ValueError("episodes must be a list")
    episodes: list[TransferEpisode] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        episode = TransferEpisode.from_dict(item, index=index)
        if episode.steps:
            episodes.append(episode)
    if not episodes:
        raise ValueError("episodes is empty")
    return episodes


def episodes_to_act_payload(
    episodes: Iterable[TransferEpisode],
    *,
    chunk_size: int = 10,
    action_labels: list[str] | None = None,
    min_steps_per_episode: int = 1,
    include_images: bool = True,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    labels = action_labels or list(DEFAULT_ACTION_LABELS)
    if not labels:
        raise ValueError("action_labels must not be empty")

    states: list[list[float]] = []
    env_states: list[list[float]] = []
    actions: list[list[list[float]]] = []
    action_is_pad: list[list[int]] = []
    images: list[list[list[str]]] = []
    robots: list[str] = []
    teleoperators: list[str] = []
    episode_refs: list[dict[str, Any]] = []
    domain_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    used_images = False
    expected_action_dim: int | None = None

    for episode in episodes:
        if len(episode.steps) < min_steps_per_episode:
            continue
        domain_counter[episode.source_domain.value] += 1
        mode_counter[episode.transfer_mode.value] += 1

        for start in range(0, len(episode.steps), chunk_size):
            chunk_steps = episode.steps[start : start + chunk_size]
            first_step = chunk_steps[0]

            states.append(list(first_step.observation.state))
            env_states.append(list(first_step.observation.environment_state))

            action_chunk: list[list[float]] = []
            pad_chunk: list[int] = []
            image_chunk: list[list[str]] = []

            for step in chunk_steps:
                action_vec = _action_to_vector(step, labels)
                if expected_action_dim is None:
                    expected_action_dim = len(action_vec)
                if len(action_vec) != expected_action_dim:
                    raise ValueError("all action vectors must have the same length")
                action_chunk.append(action_vec)
                pad_chunk.append(0)
                cameras = [frame.data_url for frame in step.observation.cameras if frame.data_url]
                if cameras:
                    used_images = True
                image_chunk.append(cameras or [""])

            if expected_action_dim is None:
                expected_action_dim = len(labels)
            while len(action_chunk) < chunk_size:
                action_chunk.append([0.0] * expected_action_dim)
                pad_chunk.append(1)
                image_chunk.append([""])

            actions.append(action_chunk)
            action_is_pad.append(pad_chunk)
            images.append(image_chunk)
            robots.append(episode.robot_id or "")
            teleoperators.append(episode.teleoperator_id or "")
            episode_refs.append(
                {
                    "episode_id": episode.episode_id,
                    "source_domain": episode.source_domain.value,
                    "transfer_mode": episode.transfer_mode.value,
                    "step_offset": start,
                    "chunk_size": chunk_size,
                }
            )

    if not actions:
        raise ValueError("no valid chunks were generated from episodes")

    payload: dict[str, Any] = {
        "states": states,
        "env_states": env_states,
        ACTION: actions,
        "action_is_pad": action_is_pad,
        ROBOTS: robots,
        TELEOPERATORS: teleoperators,
        "meta": {
            "schema": "aka_sim.transfer_episode.v1",
            "action_labels": labels,
            "transfer_modes": dict(mode_counter),
            "source_domains": dict(domain_counter),
            "episode_refs": episode_refs,
            "chunk_size": chunk_size,
            "min_steps_per_episode": min_steps_per_episode,
        },
    }
    if include_images and used_images:
        payload[OBS_IMAGES] = images
    return payload


def episode_step_to_infer_payload(step: TransferStep) -> dict[str, Any]:
    payload = {
        "state": list(step.observation.state),
        "env_state": list(step.observation.environment_state),
    }
    cameras = [frame.data_url for frame in step.observation.cameras if frame.data_url]
    if cameras:
        payload["images"] = cameras
    return payload


def _action_to_vector(step: TransferStep, labels: list[str]) -> list[float]:
    if step.action.vector:
        return [float(v) for v in step.action.vector]
    vector = [0.0] * len(labels)
    command = (step.action.command or "").strip().lower()
    if command and command in labels:
        vector[labels.index(command)] = 1.0
        return vector
    throttle = step.action.throttle
    steering = step.action.steering
    if throttle is not None or steering is not None:
        move = float(throttle or 0.0)
        turn = float(steering or 0.0)
        if abs(move) >= abs(turn):
            command = "up" if move >= 0 else "down"
        else:
            command = "right" if turn >= 0 else "left"
        if command in labels:
            vector[labels.index(command)] = 1.0
            return vector
    vector[-1] = 1.0
    return vector
