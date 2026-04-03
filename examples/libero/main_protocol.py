"""Libero evaluation using TCP+Protobuf protocol.

Two payload modes (--preprocess {client,server}):

  client (default): This process does pre/post like openpi pi05 websocket path.
    - Images: float32 NCHW (1,3,224,224), resize_with_pad, values in [-1, 1].
    - Prompt: int32 PaliGemma tokens, shape (1, max_token_len).
    - State: quantile (or z-score) norm on 8-d, zero-pad to 32, float64 (1, 32) on wire.
    - Post: optional action unnormalize + slice to 7-D.

  server: Remote does tokenize / norm / pad / action denorm; this side sends raw-ish tensors.
    - Images: uint8 NCHW (1,3,224,224), resize_with_pad only (no [-1,1] scaling).
    - Prompt: UTF-8 task language string (protobuf STRING).
    - State: raw 8-d Libero state, float64, shape (1, 8), no padding.
    - Post: usually --no-unnormalize-actions if the server already returns env actions.

Use --no-use-quantile-norm for z-score state/actions in client mode only.

Openpi PadStatesAndActions: pad_to_dim(..., value=0.0) — padded state dimensions are zeros.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
from typing import Literal

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
MAX_TOKEN_LEN_DEFAULT = 200
IMAGE_SIZE_DEFAULT = 224

DTYPES_CLIENT = {
    "images": msg_pb2.Tensor.FLOAT32,
    "prompt": msg_pb2.Tensor.INT32,
    "state": msg_pb2.Tensor.FLOAT64,
}

DTYPES_SERVER = {
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

    image_size: int = IMAGE_SIZE_DEFAULT
    max_token_len: int = MAX_TOKEN_LEN_DEFAULT
    discrete_state_in_prompt: bool = False
    use_quantile_norm: bool = True
    preprocess: Literal["client", "server"] = "client"


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


def _load_norm_stats_entry(path: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["norm_stats"][key]
    mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float64).reshape(-1)
    q01 = np.asarray(data["q01"], dtype=np.float64).reshape(-1) if data.get("q01") is not None else None
    q99 = np.asarray(data["q99"], dtype=np.float64).reshape(-1) if data.get("q99") is not None else None
    return mean, std, q01, q99


def _normalize_state_zscore_pad_32(raw8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    m = mean[:LIBERO_STATE_RAW_DIM]
    s = std[:LIBERO_STATE_RAW_DIM]
    normed = (raw8 - m) / (s + 1e-6)
    out = np.zeros(MODEL_STATE_DIM, dtype=np.float32)
    out[:LIBERO_STATE_RAW_DIM] = normed.astype(np.float32)
    return out


def _normalize_state_quantile_pad_32(raw8: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """openpi.transforms.Normalize._normalize_quantile on first 8 dims, then zero-pad to 32."""
    q1 = q01[:LIBERO_STATE_RAW_DIM]
    q9 = q99[:LIBERO_STATE_RAW_DIM]
    normed = (raw8 - q1) / (q9 - q1 + 1e-6) * 2.0 - 1.0
    out = np.zeros(MODEL_STATE_DIM, dtype=np.float32)
    out[:LIBERO_STATE_RAW_DIM] = normed.astype(np.float32)
    return out


def _state_tensor_for_server(normed32: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(normed32.astype(np.float64).reshape(1, MODEL_STATE_DIM))


def _preprocess_image_server(raw_hwc_u8: np.ndarray, size: int) -> np.ndarray:
    x = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw_hwc_u8, size, size))
    x = x.astype(np.float32) / 255.0 * 2.0 - 1.0
    chw = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
    return np.expand_dims(chw, axis=0)


def _preprocess_image_server_uint8_nchw(raw_hwc_u8: np.ndarray, size: int) -> np.ndarray:
    """Resize+pad only; uint8 NCHW (1,3,H,W) for remote preprocessing."""
    x = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw_hwc_u8, size, size))
    chw = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
    return np.expand_dims(chw, axis=0).astype(np.uint8)


def _state_tensor_raw8_for_server(raw8: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw8.astype(np.float64).reshape(1, LIBERO_STATE_RAW_DIM))


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
    return toks.reshape(1, max_len)


def _extract_raw_actions(result: dict) -> np.ndarray:
    for key in ("state", "prompt"):
        val = result.get(key)
        if val is not None and isinstance(val, np.ndarray) and val.ndim >= 2:
            if val.ndim == 3:
                val = val[0]
            return np.asarray(val, dtype=np.float64)[:LIBERO_ACTION_HORIZON, :]
    raise RuntimeError(f"No action tensor in response. Keys: {list(result.keys())}")


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


def _unnormalize_actions_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """openpi.transforms.Unnormalize._unnormalize_quantile."""
    q01 = np.asarray(q01, dtype=np.float64).reshape(-1)
    q99 = np.asarray(q99, dtype=np.float64).reshape(-1)
    dim = q01.shape[-1]
    dlast = x.shape[-1]
    if dim < dlast:
        left = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        right = x[..., dim:]
        return np.concatenate([left, right], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _postprocess_actions(
    raw: np.ndarray,
    *,
    do_unnorm: bool,
    use_quantile: bool,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_q01: np.ndarray | None,
    action_q99: np.ndarray | None,
) -> np.ndarray:
    if do_unnorm:
        if use_quantile:
            if action_q01 is None or action_q99 is None:
                raise ValueError("norm_stats.actions missing q01/q99; cannot use quantile unnormalize")
            raw = _unnormalize_actions_quantile(raw, action_q01, action_q99)
        else:
            raw = _unnormalize_actions_zscore(raw, action_mean, action_std)
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
    state_mean, state_std, state_q01, state_q99 = _load_norm_stats_entry(str(stats_path), "state")
    action_mean, action_std, action_q01, action_q99 = _load_norm_stats_entry(str(stats_path), "actions")
    logging.info(f"Loaded norm_stats from {stats_path}")
    logging.info(f"Protocol preprocess mode: {args.preprocess!r} (client=tokenize+norm here; server=raw images, str prompt, 8-d state)")
    if args.preprocess == "server" and args.discrete_state_in_prompt:
        logging.warning("discrete_state_in_prompt is ignored in server preprocess mode (prompt is plain text).")

    if args.use_quantile_norm and args.preprocess == "client":
        if state_q01 is None or state_q99 is None:
            raise ValueError("norm_stats.state must include q01 and q99 for --use-quantile-norm (default) in client mode")
        logging.info("State normalize (client): quantile (pi05-style)")
    elif args.preprocess == "client":
        logging.info("State normalize (client): z-score (mean/std)")

    if args.unnormalize_actions and args.use_quantile_norm:
        if action_q01 is None or action_q99 is None:
            raise ValueError("norm_stats.actions must include q01 and q99 for quantile action unnormalize")

    tokenizer: _pali_tokenizer_mod.PaligemmaTokenizer | None = None
    if args.preprocess == "client":
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
                        raw8 = _build_raw_state_8(obs)
                        if args.preprocess == "client":
                            assert tokenizer is not None
                            img_f = _preprocess_image_server(raw_img, args.image_size)
                            wrist_f = _preprocess_image_server(raw_wrist, args.image_size)
                            dummy_f = np.zeros_like(img_f, dtype=np.float32)
                            if args.use_quantile_norm:
                                normed32 = _normalize_state_quantile_pad_32(raw8, state_q01, state_q99)
                            else:
                                normed32 = _normalize_state_zscore_pad_32(raw8, state_mean, state_std)
                            prompt_out = _tokenize_prompt(
                                tokenizer,
                                str(task_description),
                                args.max_token_len,
                                discrete_state=args.discrete_state_in_prompt,
                                normed_state_32=normed32,
                            )
                            state_send = _state_tensor_for_server(normed32)
                            dtypes_send = DTYPES_CLIENT
                        else:
                            img_f = _preprocess_image_server_uint8_nchw(raw_img, args.image_size)
                            wrist_f = _preprocess_image_server_uint8_nchw(raw_wrist, args.image_size)
                            dummy_f = np.zeros_like(img_f, dtype=np.uint8)
                            prompt_out = str(task_description).strip()
                            state_send = _state_tensor_raw8_for_server(raw8)
                            dtypes_send = DTYPES_SERVER

                        observation = {
                            "images": {
                                "cam_high": img_f,
                                "cam_left_wrist": wrist_f,
                                "cam_right_wrist": dummy_f,
                            },
                            "prompt": prompt_out,
                            "state": state_send,
                        }

                        server.send(observation, dtypes_send, reset=is_first_infer)
                        is_first_infer = False

                        result = server.receive()
                        if result is None:
                            logging.error("Failed to receive from inference client")
                            break

                        raw_actions = _extract_raw_actions(result)
                        action_chunk = _postprocess_actions(
                            raw_actions,
                            do_unnorm=args.unnormalize_actions,
                            use_quantile=args.use_quantile_norm,
                            action_mean=action_mean,
                            action_std=action_std,
                            action_q01=action_q01,
                            action_q99=action_q99,
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
