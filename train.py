import os
import argparse
import functools
from datetime import datetime
from typing import Any, Dict, Sequence, Tuple, Union

# MuJoCo & Brax Imports
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac

# JAX & Orbax Imports
import jax
from jax import numpy as jp
import numpy as np
from orbax import checkpoint as ocp
from flax.training import orbax_utils
import matplotlib.pyplot as plt


def progress(num_steps, metrics):
    print(f"Step {num_steps}: reward={metrics['eval/episode_reward']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with a specific environment.')
    parser.add_argument('--env_name', type=str, default='PandaPickCubeOrientation', help='Name of the environment')
    args = parser.parse_args()

    env_name = args.env_name
    ckpt_dir = os.path.abspath('./model_checkpoints/'+ env_name) 
    
    print(f"Environment: {env_name}")
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    ppo_params = manipulation_params.brax_ppo_config(env_name)
    
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks

    if "network_factory" in ppo_training_params:
        factory_args = ppo_training_params.pop("network_factory")
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **factory_args
        )

    # --- 3.5 CHECKPOINTING (ARA KAYIT) ---
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    class CheckpointCallback:
        def __init__(self, interval, directory, checkpointer):
            self.interval = interval
            self.directory = directory
            self.checkpointer = checkpointer
            self.last_step = 0
            self.last_saved_time = datetime.now()

        def __call__(self, step, make_policy, params):
            # İlk adımda (0) veya interval geçince kaydet
            if step == 0 or step >= self.last_step + self.interval:
                save_path = os.path.join(self.directory, f'{step}')
                print(f"Saving checkpoint at step {step} to {save_path}")
                save_args = orbax_utils.save_args_from_target(params)
                self.checkpointer.save(save_path, params, save_args=save_args)
                self.last_step = step
                
                # Süre bilgisini de yazalım
                now = datetime.now()
                print(f"Time since last save: {now - self.last_saved_time}")
                self.last_saved_time = now

    # Her 500k adımda bir kaydet
    save_checkpoint = CheckpointCallback(
        interval=500_000, 
        directory=ckpt_dir, 
        checkpointer=orbax_checkpointer
    )

    train_fn = functools.partial(
        ppo.train, 
        **ppo_training_params, 
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=save_checkpoint,
        seed=1
    )

    print("Eğitim başlıyor... (Bu işlem GPU hızına göre zaman alabilir)")
    start_time = datetime.now()
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    end_time = datetime.now()
    print(f"Eğitim tamamlandı! Süre: {end_time - start_time}")

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    save_path = os.path.join(ckpt_dir, 'final_model')
    
    print(f"Model kaydediliyor: {save_path}")
    save_args = orbax_utils.save_args_from_target(params)
    orbax_checkpointer.save(save_path, params, save_args=save_args)
    print("Kayıt başarılı.")

    try:
        pass 
    except Exception as e:
        print(f"Grafik çizilemedi: {e}")