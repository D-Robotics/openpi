"""Libero evaluation using TCP+Protobuf protocol.

Client-side preprocessing for inference server:
  - Images: float32 CHW (3,224,224), resize_with_pad, values in [-1, 1].
  - Prompt: int32 PaliGemma tokens, shape (max_token_len,), default (200,).
  - State: z-score normalize 8-d libero state (norm_stats), pad to 32, float16, shape (10, 32).

Post-processing: optional action z-score unnormalize + slice to 7-D (--no-unnormalize-actions if server
already returns physical actions).
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
from openpi_client import image_tools
import tqdm
import tyro

_OPENPI_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if _OPENPI_SRC.is_dir() and str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))

from openpi.models import tokenizer as _pali_tokenizer_mod

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "protocol"))
import msg_pb2
from net import Server

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_HORIZON = 10
LIBERO_STATE_RAW_DIM = 8
MODEL_STATE_DIM = 32
STATE_TIME_EXPAND = 10
MAX_TOKEN_LEN_DEFAULT = 200
IMAGE_SIZE_DEFAULT = 224

DTYPES = {
    "images": msg_pb2.Tensor.FLOAT32,
    "prompt": msg_pb2.Tensor.INT32,
    "state": msg_pb2.Tensor.FP16,
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

    image_size: int = IMAGE_SIZE_DEFAULT
    max_token_len: int = MAX_TOKEN_LEN_DEFAULT
    discrete_state_in_prompt: bool = False


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _build_raw_state_8(obs: dict) -> np.ndarray:
    return np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )).astype(np.float64)


def _load_state_norm_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["norm_stats"]["state"]
    mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float64).reshape(-1)
    return mean, std


def _normalize_state_pad_32(raw8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    m = mean[:LIBERO_STATE_RAW_DIM]
    s = std[:LIBERO_STATE_RAW_DIM]
    normed = (raw8 - m) / (s + 1e-6)
    out = np.zeros(MODEL_STATE_DIM, dtype=np.float32)
    out[:LIBERO_STATE_RAW_DIM] = normed.astype(np.float32)
    return out


def _state_tensor_for_server(normed32: np.ndarray) -> np.ndarray:
    v = normed32.astype(np.float16).reshape(1, MODEL_STATE_DIM)
    return np.broadcast_to(v, (STATE_TIME_EXPAND, MODEL_STATE_DIM)).copy()


def _preprocess_image_server(raw_hwc_u8: np.ndarray, size: int) -> np.ndarray:
    x = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw_hwc_u8, size, size))
    x = x.astype(np.float32) / 255.0 * 2.0 - 1.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def _preprocess_image_replay(raw_hwc_u8: np.ndarray, size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(raw_hwc_u8, size, size))


def _tokenize_prompt(
    tokenizer: _pali_tokenizer_mod.PaligemmaTokenizer,
    prompt: str,
    max_len: int,
    *,
    discrete_state: bool,
    normed_state_32: np.ndarray | None,
) -> np.ndarray:
    st = None if not discrete_state else np.asarray(normed_state_32, dtype=np.float64)
    tokens, _mask = tokenizer.tokenize(prompt.strip(), st)
    toks = np.asarray(tokens, dtype=np.int32).reshape(-1)
    if toks.size != max_len:
        raise RuntimeError(f"Expected {max_len} tokens, got {toks.size}")
    return toks


def _extract_raw_actions(result: dict) -> np.ndarray:
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
    if mean.size < action_dim:
        mean = np.pad(mean, (0, action_dim - mean.size))
        std = np.pad(std, (0, action_dim - std.size), constant_values=1.0)
    elif mean.size > action_dim:
        mean = mean[:action_dim]
        std = std[:action_dim]
    return mean, std


def _unnormalize_actions_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
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

    stats_path = pathlib.Path(args.norm_stats_path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"norm_stats not found: {stats_path.resolve()}")
    state_mean, state_std = _load_state_norm_stats(str(stats_path))
    logging.info(f"Loaded state norm_stats from {stats_path}")

    action_mean, action_std = (np.zeros(0), np.zeros(0))
    if args.unnormalize_actions:
        action_mean, action_std = _load_action_norm_stats(str(stats_path))
        logging.info("Loaded action norm_stats for unnormalize")

    logging.info(
        "Loading PaliGemma tokenizer (first run may download from gs://big_vision/..., can take minutes)..."
    )
    tokenizer = _pali_tokenizer_mod.PaligemmaTokenizer(max_len=args.max_token_len)
    logging.info("Tokenizer ready.")

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
                    replay_images.append(_preprocess_image_replay(raw_img, args.image_size))

                    if not action_plan:
                        img_f = _preprocess_image_server(raw_img, args.image_size)
                        wrist_f = _preprocess_image_server(raw_wrist, args.image_size)
                        dummy_f = np.zeros_like(img_f, dtype=np.float32)

                        raw8 = _build_raw_state_8(obs)
                        normed32 = _normalize_state_pad_32(raw8, state_mean, state_std)
                        prompt_tokens = _tokenize_prompt(
                            tokenizer,
                            str(task_description),
                            args.max_token_len,
                            discrete_state=args.discrete_state_in_prompt,
                            normed_state_32=normed32,
                        )
                        state_send = _state_tensor_for_server(normed32)

                        observation = {
                            "images": {
                                "cam_high": img_f,
                                "cam_left_wrist": wrist_f,
                                "cam_right_wrist": dummy_f,
                            },
                            "prompt": prompt_tokens,
                            "state": state_send,
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
