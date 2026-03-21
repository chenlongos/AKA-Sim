from __future__ import annotations

import argparse
import contextlib
import time

from real_car_common import (
    ClientConfig,
    backend_post_json,
    get_observation,
    open_camera,
    send_robot_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实小车在线视觉推理客户端")
    parser.add_argument("--backend-url", default="http://127.0.0.1:5000", help="AKA-Sim 后端地址")
    parser.add_argument("--robot-base-url", default=None, help="真实小车控制端地址，例如 http://192.168.1.101")
    parser.add_argument("--telemetry-url", default=None, help="状态观测 JSON 接口，需返回 state/environment_state")
    parser.add_argument("--model-id", default=None, help="已训练模型 ID")
    parser.add_argument("--model-path", default=None, help="模型 checkpoint 路径")
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--env-dim", type=int, default=7)
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--action-time", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="只请求推理，不向真实小车发动作")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_id and not args.model_path:
        raise SystemExit("必须提供 --model-id 或 --model-path")
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

    start_payload = {}
    if args.model_id:
        start_payload["model_id"] = args.model_id
    if args.model_path:
        start_payload["model_path"] = args.model_path
    start_info = backend_post_json(config.backend_url, "/api/infer/start", start_payload, timeout=config.timeout)
    print(f"推理启动: {start_info}")

    interval = 1.0 / max(args.hz, 0.1)
    with open_camera(config) as camera:
        try:
            while True:
                frame_start = time.time()
                image_data_url = camera.read_data_url()
                observation = get_observation(
                    telemetry_url=config.telemetry_url,
                    state_dim=config.state_dim,
                    env_dim=config.env_dim,
                    image_data_url=image_data_url,
                )
                infer_payload = {
                    "observation": {
                        "state": observation["state"],
                        "environment_state": observation["environment_state"],
                        "images": [image_data_url],
                    }
                }
                result = backend_post_json(config.backend_url, "/api/infer/step", infer_payload, timeout=config.timeout)
                action = str(result.get("action") or "stop")
                send_robot_action(
                    robot_base_url=config.robot_base_url,
                    action=action,
                    speed=config.speed,
                    action_time=config.action_time,
                    timeout=config.timeout,
                    dry_run=config.dry_run,
                )
                print(f"action={action:<5} vector={result.get('action_vector')}")
                elapsed = time.time() - frame_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        except KeyboardInterrupt:
            print("收到中断，准备停止小车并退出")
        finally:
            with contextlib.suppress(Exception):
                send_robot_action(
                    robot_base_url=config.robot_base_url,
                    action="stop",
                    speed=config.speed,
                    action_time=config.action_time,
                    timeout=config.timeout,
                    dry_run=config.dry_run,
                )
            with contextlib.suppress(Exception):
                backend_post_json(config.backend_url, "/api/infer/stop", {}, timeout=config.timeout)


if __name__ == "__main__":
    main()
