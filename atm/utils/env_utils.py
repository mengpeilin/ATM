from collections import OrderedDict
from collections.abc import Iterable


def build_env(img_size, env_type, env_meta_fn=None, env_name=None, task_name=None,
              render_gpu_ids=-1, vec_env_num=1, seed=0, env_idx_start_end=None, **kwargs):
    """
    Build the rollout environment.
    Args:
        img_size: The resolution of the pixel observation.
        env_type: The type of environment benchmark. Choices: ["libero"].
        env_meta_fn: The path to robommimic meta data, which is used to specify the robomimic environments.
        env_name: The name to specify the environments.
        obs_types: The observation types in the returned obs dict in Robomimic
        render_gpu_ids:  The available GPU ids for rendering the images
        vec_env_num: The number of parallel environments
        seed: The random seed environment initialization.

    Returns:
        env: An OrderedDict mapping a description to (env_idx, gym-like environment). It is empty
            when `env_name` is empty, which is how training runs without any simulator.
    """
    if not env_name:
        # No rollout environments requested (e.g. RoboTwin, whose evaluation is driven by RoboTwin's
        # own evaluator). Return early so that the simulator backends are never imported.
        return OrderedDict()

    if isinstance(img_size, Iterable):
        assert len(img_size) == 2
        img_h = img_size[0]
        img_w = img_size[1]
    else:
        img_h = img_w = img_size

    if env_type.lower() == "libero":
        # Imported lazily so that libero / robosuite are only needed when actually rolling out
        # LIBERO environments.
        from atm.utils.libero_env_utils import build_libero_env

        return build_libero_env(img_h, img_w, env_meta_fn=env_meta_fn, env_name=env_name,
                                task_name=task_name, render_gpu_ids=render_gpu_ids,
                                vec_env_num=vec_env_num, seed=seed,
                                env_idx_start_end=env_idx_start_end)
    else:
        raise ValueError(f"Environment {env_type} is not supported!")
