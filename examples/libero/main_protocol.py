"""Libero evaluation using TCP+Protobuf protocol.

Client: state pad, image rotate + HWC→CHW, optional action unnormalize (openpi z-score).
Default norm_stats: examples/libero/norm_stats.json (8-d state, 7-d actions).
If server returns [H,32] normalized actions, stats are auto-padded to 32 like openpi.
If the server already returns physical actions, pass --no-unnormalize-actions.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "protocol"))
import msg_pb2
from net import Server

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_HORIZON = 10
LIBERO_STATE_PAD_TO = 8

DTYPES = {
    "images": msg_pb2.Tensor.UINT8,
    "prompt": msg_pb2.Tensor.STRING,
    "state": msg_pb2.Tensor.FLOAT64,
}


def _default_norm_stats_path() -> str:
    return str(pathlib.Path(__file__).resolve().parent / "norm_stats.json")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8888
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    video_out_path: str = "data/libero/videos"
    seed: int = 7

    unnormalize_actions: bool = True
    norm_stats_path: str = dataclasses.field(default_factory=_default_norm_stats_path)


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


def _build_state(obs: dict) -> np.ndarray:
    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    if LIBERO_STATE_PAD_TO > len(state):
        state = np.pad(state, (0, LIBERO_STATE_PAD_TO - len(state)))
    return state


def _extract_raw_actions(result: dict) -> np.ndarray:
    """Action tensor from server (e.g. [10, 32] or [10, 7]); horizon truncated to 10."""
    for key in ("state", "prompt"):
        val = result.get(key)
        if val is not None and isinstance(val, np.ndarray) and val.ndim >= 2:
            if val.ndim == 3:
                val = val[0]
            return np.asarray(val, dtype=np.float64)[:LIBERO_ACTION_HORIZON, :]
    raise RuntimeError(f"No action tensor in response. Keys: {list(result.keys())}")


def _load_action_norm_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["norm_stats"]["actions"]
    mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float64).reshape(-1)
    return mean, std


def _pad_stats_to_action_dim(mean: np.ndarray, std: np.ndarray, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Match openpi transforms.Unnormalize._unnormalize: pad mean with 0, std with 1 if short; truncate if long."""
    if mean.size < action_dim:
        mean = np.pad(mean, (0, action_dim - mean.size))
        std = np.pad(std, (0, action_dim - std.size), constant_values=1.0)
    elif mean.size > action_dim:
        mean = mean[:action_dim]
        std = std[:action_dim]
    return mean, std


def _unnormalize_actions_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Same as openpi.transforms.Unnormalize._unnormalize for the actions key."""
    d = x.shape[-1]
    mean, std = _pad_stats_to_action_dim(mean, std, d)
    return x * (std + 1e-6) + mean


def _postprocess_actions(raw: np.ndarray, mean: np.ndarray, std: np.ndarray, *, do_unnorm: bool) -> np.ndarray:
    if do_unnorm:
        raw = _unnormalize_actions_zscore(raw, mean, std)
    return raw[:, :LIBERO_ACTION_DIM]


def _get_max_steps(task_suite_name: str) -> int:
    limits = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in limits:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return limits[task_suite_name]


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    action_mean, action_std = (np.zeros(0), np.zeros(0))
    if args.unnormalize_actions:
        p = pathlib.Path(args.norm_stats_path)
        if not p.is_file():
            raise FileNotFoundError(f"norm_stats not found: {p.resolve()}")
        action_mean, action_std = _load_action_norm_stats(str(p))
        logging.info(f"Loaded action norm_stats from {p}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = _get_max_steps(args.task_suite_name)

    server = Server(port=args.port)
    logging.info(f"Waiting for inference client on port {args.port}...")
    server.connect()
    logging.info("Inference client connected.")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            is_first_infer = True

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    raw_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    raw_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(raw_img.copy())

                    if not action_plan:
                        img_chw = _hwc_to_chw(raw_img)
                        wrist_chw = _hwc_to_chw(raw_wrist)
                        dummy_chw = np.zeros_like(img_chw)

                        observation = {
                            "images": {
                                "cam_high": img_chw,
                                "cam_left_wrist": wrist_chw,
                                "cam_right_wrist": dummy_chw,
                            },
                            "prompt": np.array(str(task_description)),
                            "state": _build_state(obs),
                        }

                        server.send(observation, DTYPES, reset=is_first_infer)
                        is_first_infer = False

                        result = server.receive()
                        if result is None:
                            logging.error("Failed to receive from inference client")
                            break

                        raw_actions = _extract_raw_actions(result)
                        action_chunk = _postprocess_actions(
                            raw_actions,
                            action_mean,
                            action_std,
                            do_unnorm=args.unnormalize_actions,
                        )

                        assert len(action_chunk) >= args.replan_steps, (
                            f"Want {args.replan_steps} steps, got {len(action_chunk)}."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Final success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
