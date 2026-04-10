import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _install_egl_cleanup_guard() -> None:
    """Guard against robosuite EGL cleanup exceptions during Python finalization."""
    try:
        from robosuite.renderers.context import egl_context as _egl_context
        from robosuite.utils import binding_utils as _binding_utils
    except Exception:
        return

    if not getattr(_egl_context.EGLGLContext.free, "_openpi_safe_cleanup", False):
        original_free = _egl_context.EGLGLContext.free

        def _safe_free(self):
            try:
                return original_free(self)
            except Exception as e:
                logging.debug("Ignoring EGL cleanup exception in EGLGLContext.free: %s", e)
                return None

        _safe_free._openpi_safe_cleanup = True  # type: ignore[attr-defined]
        _egl_context.EGLGLContext.free = _safe_free

    if not getattr(_binding_utils.MjRenderContext.__del__, "_openpi_safe_cleanup", False):
        original_del = _binding_utils.MjRenderContext.__del__

        def _safe_del(self):
            try:
                return original_del(self)
            except Exception as e:
                logging.debug("Ignoring EGL cleanup exception in MjRenderContext.__del__: %s", e)
                return None

        _safe_del._openpi_safe_cleanup = True  # type: ignore[attr-defined]
        _binding_utils.MjRenderContext.__del__ = _safe_del


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    rollout_save_path: str = ""  # If set, dump per-episode rollouts (videos + state/action npz) for later LeRobot conversion

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    _install_egl_cleanup_guard()

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    rollout_save_root = pathlib.Path(args.rollout_save_path) if args.rollout_save_path else None
    if rollout_save_root is not None:
        rollout_save_root.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        num_init_states = len(initial_states)
        if num_init_states == 0:
            raise ValueError(f"No initial states found for task_id={task_id} in suite={args.task_suite_name}")
        if args.num_trials_per_task > num_init_states:
            logging.warning(
                "num_trials_per_task=%d exceeds available initial states=%d for task_id=%d. "
                "Cycling initial states with modulo indexing.",
                args.num_trials_per_task,
                num_init_states,
                task_id,
            )

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                init_state_idx = episode_idx % num_init_states
                obs = env.set_init_state(initial_states[init_state_idx])

                # Setup
                done = False
                t = 0
                replay_images = []
                ep_images_256 = []
                ep_wrist_256 = []
                ep_states = []
                ep_actions = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img_256 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img_256 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img_256, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img_256, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": state,
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Record per-step rollout data (paired with the action we are about to execute)
                        if rollout_save_root is not None:
                            ep_images_256.append(img_256)
                            ep_wrist_256.append(wrist_img_256)
                            ep_states.append(state)
                            ep_actions.append(np.asarray(action, dtype=np.float32))

                        # Execute action in environment
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

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                # Dump per-episode raw rollout (256x256 videos + state/action npz) for LeRobot conversion.
                if rollout_save_root is not None and len(ep_actions) > 0:
                    ep_dir = rollout_save_root / f"task{task_id:02d}_{task_segment}" / f"ep{episode_idx:03d}_{suffix}"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(ep_dir / "agentview.mp4", ep_images_256, fps=10)
                    imageio.mimwrite(ep_dir / "wrist.mp4", ep_wrist_256, fps=10)
                    np.savez(
                        ep_dir / "data.npz",
                        state=np.stack(ep_states),
                        actions=np.stack(ep_actions),
                        task=str(task_description),
                        success=bool(done),
                        task_suite=str(args.task_suite_name),
                        task_id=int(task_id),
                        episode_idx=int(episode_idx),
                    )

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        finally:
            _close_libero_env(env)

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _close_libero_env(env) -> None:
    """Close robosuite/MuJoCo EGL resources before Python finalizers run."""
    try:
        env.close()
    except Exception as e:
        logging.warning("Ignoring exception while closing LIBERO environment: %s", e)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": False,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
