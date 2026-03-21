from __future__ import annotations

import argparse
import contextlib
import select
import sys
import termios
import time
import tty
from typing import Any

from real_car_common import (
    ACTION_KEYS,
    ClientConfig,
    backend_post_json,
    get_observation,
    open_camera,
    print_json,
    send_robot_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集真实小车 RealToReal / SimToReal episode 并上传到 AKA-Sim")
    parser.add_argument("--backend-url", default="http://127.0.0.1:5000", help="AKA-Sim 后端地址")
    parser.add_argument("--robot-base-url", default=None, help="真实小车控制端地址，例如 http://192.168.1.101")
    parser.add_argument("--telemetry-url", default=None, help="状态观测 JSON 接口，需返回 state/environment_state")
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--env-dim", type=int, default=7)
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--hz", type=float, default=5.0, help="采样频率")
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--action-time", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--robot-id", default="car_01")
    parser.add_argument("--teleoperator-id", default="operator_a")
    parser.add_argument("--source-domain", default="real", choices=["real", "sim"])
    parser.add_argument("--transfer-mode", default="real_to_real", choices=["real_to_real", "sim_to_real"])
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="只采集和上传，不向真实小车发控制")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClientConfig(
        backend_url=args.backend_url,
        robot_base_url=args.robot_base_url,
        telemetry_url=args.telemetry_url,
        state_dim=args.state_dim,
        env_dim=args.env_dim,
        camera_device=args.camera_device,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        speed=args.speed,
        action_time=args.action_time,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    interval = 1.0 / max(args.hz, 0.1)

    episodes: list[dict[str, Any]] = []
    current_steps: list[dict[str, Any]] = []
    current_action = "stop"
    last_sent_action = None

    print("控制键: w/a/s/d, 空格=stop, n=结束当前 episode, u=上传, q=退出")
    print("开始采集后会持续抓图并记录 observation/action。")

    with open_camera(config) as camera, raw_terminal():
        try:
            while True:
                key = read_key_nonblocking(timeout=interval)
                if key in ACTION_KEYS:
                    current_action = ACTION_KEYS[key]
                    if current_action != last_sent_action:
                        send_robot_action(
                            robot_base_url=config.robot_base_url,
                            action=current_action,
                            speed=config.speed,
                            action_time=config.action_time,
                            timeout=config.timeout,
                            dry_run=config.dry_run,
                        )
                        last_sent_action = current_action
                elif key == "n":
                    flush_episode(current_steps, episodes, args)
                    current_steps = []
                    current_action = "stop"
                    print(f"episode 已切换，总数={len(episodes)}")
                    continue
                elif key == "u":
                    flush_episode(current_steps, episodes, args)
                    current_steps = []
                    upload_episodes(episodes, args, timeout=config.timeout)
                    continue
                elif key == "q":
                    flush_episode(current_steps, episodes, args)
                    current_steps = []
                    if episodes:
                        upload_episodes(episodes, args, timeout=config.timeout)
                    break

                image_data_url = camera.read_data_url()
                observation = get_observation(
                    telemetry_url=config.telemetry_url,
                    state_dim=config.state_dim,
                    env_dim=config.env_dim,
                    image_data_url=image_data_url,
                )
                current_steps.append(
                    {
                        "observation": observation,
                        "action": {"command": current_action},
                    }
                )
                sys.stdout.write(
                    f"\r当前动作={current_action:<5} 当前episode步数={len(current_steps):<5} 已完成episodes={len(episodes):<3}"
                )
                sys.stdout.flush()
        finally:
            print()
            with contextlib.suppress(Exception):
                send_robot_action(
                    robot_base_url=config.robot_base_url,
                    action="stop",
                    speed=config.speed,
                    action_time=config.action_time,
                    timeout=config.timeout,
                    dry_run=config.dry_run,
                )


def flush_episode(current_steps: list[dict[str, Any]], episodes: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if len(current_steps) < args.min_steps:
        if current_steps:
            print(f"忽略当前 episode，步数 {len(current_steps)} 小于 min-steps={args.min_steps}")
        return
    episodes.append(
        {
            "episode_id": f"{args.source_domain}_ep_{len(episodes) + 1:04d}",
            "source_domain": args.source_domain,
            "transfer_mode": args.transfer_mode,
            "robot_id": args.robot_id,
            "teleoperator_id": args.teleoperator_id,
            "steps": list(current_steps),
        }
    )


def upload_episodes(episodes: list[dict[str, Any]], args: argparse.Namespace, *, timeout: float) -> None:
    if not episodes:
        print("没有可上传的 episode")
        return
    payload = {
        "chunk_size": args.chunk_size,
        "min_steps_per_episode": args.min_steps,
        "include_images": True,
        "episodes": episodes,
    }
    response = backend_post_json(args.backend_url, "/api/transfer/dataset", payload, timeout=timeout)
    print("\n上传完成:")
    print_json(response)


@contextlib.contextmanager
def raw_terminal():
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def read_key_nonblocking(*, timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    char = sys.stdin.read(1)
    if not char:
        return None
    return char


if __name__ == "__main__":
    main()
