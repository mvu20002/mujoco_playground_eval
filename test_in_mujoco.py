import os
import functools
import time
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import viewer
import argparse
import imageio
from PIL import Image, ImageDraw, ImageFont

# Brax & Playground Imports
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training import types
from orbax import checkpoint as ocp

# --- RANDOMIZATION CONFIG ---
# Global flags can be defaults, but we will rely more on env-specific logic
RANDOMIZE_JOINTS_EXTREMELY = True

def randomize_panda_pick_cube(state, rng):
    """Specific randomization for PandaPickCube environments."""
    # Randomize cube position
    while 1:
        new_pos = np.array([0.0, 0.0, 0.05]) + np.random.uniform(-0.75, 0.75, size=3)
        new_pos[2] = 0.05 # Keep on table height
        # if too close to origin, skip
        if np.linalg.norm(new_pos) < 0.1:
            continue
        break
    
    q = np.array(state.data.qpos)
    qd = np.array(state.data.qvel)
    
    q[9:12] = new_pos
    qd[9:15] = 0.0 
    
    new_mjx_data = state.data.replace(
        qpos=jp.array(q),
        qvel=jp.array(qd)
    )
    return state.replace(data=new_mjx_data)

def randomize_panda_handover(state, rng):
    """Specific randomization for PandaHandOver."""
    while 1:
        new_pos = np.array([0.0, 0.0, 0.05]) + np.random.uniform(-0.2, 0.2, size=3)
        new_pos[2] = 0.05 # Keep on table height
        # if too close to origin, skip
        if np.linalg.norm(new_pos) < 0.1:
            continue
        break
    
    q = np.array(state.data.qpos)
    qd = np.array(state.data.qvel)
    
    q[16:19] = new_pos
    qd[16:19] = 0.0 
    
    new_mjx_data = state.data.replace(
        qpos=jp.array(q),
        qvel=jp.array(qd)
    )
    return state.replace(data=new_mjx_data)

def randomize_panda_open_cabinet(state, rng):
    """Specific randomization for PandaOpenCabinet."""
    # Placeholder for cabinet randomization
    # For now, maybe just keep it simple or randomize initial arm pos more?
    return state

def randomize_joints_global(state, rng):
    """Applies global joint randomization to the robot arm."""
    first_three_joints_delta = np.random.uniform(-1.6, 1.6, size=3)
    last_three_joints_delta = np.random.uniform(-0.8, 0.8, size=3)
    q = np.array(state.data.qpos)
    qd = np.array(state.data.qvel)
    
    # Assuming first 7 joints are robot arm (excluding gripper mostly or just main joints)
    # Original code processed indices 1:7 (6 joints). Panda has 7 DOFs usually.
    # Original code: q[1:7] += random_joint_delta
    # We will stick to the original logic unless generalizable.
    
    q[1:4] += first_three_joints_delta
    q[4:7] += last_three_joints_delta
    qd[1:7] = 0.0 
    
    new_mjx_data = state.data.replace(
        qpos=jp.array(q),
        qvel=jp.array(qd)
    )
    return state.replace(data=new_mjx_data)

def randomize_joints_handover(state, rng):
    """Specific randomization for PandaHandOver."""
    first_left_three_joints_delta = np.random.uniform(-1.6, 1.6, size=3)
    last_left_three_joints_delta = np.random.uniform(-0.8, 0.8, size=3)
    first_right_three_joints_delta = np.random.uniform(-1.6, 1.6, size=3)
    last_right_three_joints_delta = np.random.uniform(-0.8, 0.8, size=3)
    
    q = np.array(state.data.qpos)
    qd = np.array(state.data.qvel)
    
    q[1:4] += first_left_three_joints_delta
    q[4:7] += last_left_three_joints_delta
    q[9:12] += first_right_three_joints_delta
    q[12:15] += last_right_three_joints_delta
    qd[1:15] = 0.0
    
    new_mjx_data = state.data.replace(
        qpos=jp.array(q),
        qvel=jp.array(qd)
    )
    return state.replace(data=new_mjx_data)
    
    

def main():
    parser = argparse.ArgumentParser(description='Run MuJoCo visualization with policy.')
    parser.add_argument('--env_name', type=str, default='PandaPickCubeOrientation', help='Name of the environment')
    parser.add_argument('--policy-path', type=str, default=None, help='Path to the policy checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--randomize-joints', action='store_true', default=False, help='Apply extreme joint randomization')
    parser.add_argument('--no-randomize-joints', action='store_false', dest='randomize_joints', help='Disable extreme joint randomization')
    parser.add_argument('--randomize-domain', action='store_true', default=False, help='Apply domain/task specific randomization')
    parser.add_argument('--video_name', type=str, default=None, help='Name of the Video to save (without extension)')
    parser.add_argument('--record_attempt', type=int, default=None, help='Number of attempts to record')
    
    args = parser.parse_args()

    # Set numpy seed
    np.random.seed(args.seed)

    # --- 1. SETUP ---
    env_name = args.env_name
    
    if args.policy_path:
        model_path = args.policy_path
    else:
        ckpt_dir = os.path.abspath('./model_checkpoints/' + env_name)
        model_path = os.path.join(ckpt_dir, 'final_model')

    print(f"Loading environment: {env_name} with seed {args.seed}")
    print(f"Loading checkpoint from: {model_path}")

    env = registry.load(env_name)
    
    # --- 2. RECONSTRUCT NETWORK ---
    print("Reconstructing network...")
    ppo_params = manipulation_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)
    
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_training_params:
        factory_args = ppo_training_params.pop("network_factory")
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **factory_args
        )

    # Determine preprocessor
    normalize_observations = ppo_params.get('normalize_observations', True)
    
    if normalize_observations:
        # Define a wrapper to handle dict params (orbax restoration artifact)
        import collections
        MeanStd = collections.namedtuple('MeanStd', ['mean', 'std', 'count']) 
        
        def normalize_fn(obs, params):
            if isinstance(params, dict):
                class State:
                    pass
                state = State()
                state.mean = params['mean']
                state.std = params['std']
                params = state
            return running_statistics.normalize(obs, params)
            
        preprocess_fn = normalize_fn
    else:
        preprocess_fn = types.identity_observation_preprocessor

    # initialize network to get structure
    network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn
    )

    # --- 3. LOAD CHECKPOINT ---
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")
            
        params = orbax_checkpointer.restore(model_path)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # --- 4. PREPARE INFERENCE ---
    make_inference_fn = ppo_networks.make_inference_fn(network)
    inference_fn = make_inference_fn(params)
    inference_fn = jax.jit(inference_fn)
    
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # --- 5. VISUALIZATION LOOP ---
    print("Launching MuJoCo...", flush=True)
    
    # Get MuJoCo model and data
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    if args.video_name:
        # --- HEADLESS RECORDING MODE ---
        print(f"Headless Mode Enabled. Recording to Video: {args.video_name}", flush=True)
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
        
        # Setup Video output path
        results_dir = os.path.abspath(f'./results/{env_name}')
        os.makedirs(results_dir, exist_ok=True)
        video_path = os.path.join(results_dir, f'{args.video_name}.mp4')
        fps = int(1.0 / env.dt)
        
        # Open video writer
        # quality=5 is default, decent. quality=3 is "minimal" but decent.
        # macro_block_size=None allows arbitrary resolution
        writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=6, pixelformat='yuv420p', macro_block_size=None)
        
        # JIT Warmup (Simplified for headless)
        rng = jax.random.PRNGKey(args.seed)
        state = reset_fn(rng)
        
        # Sync
        mj_data.qpos[:] = np.array(state.data.qpos)
        mj_data.qvel[:] = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        # Compile Step
        rng, subkey = jax.random.split(rng)
        act_rng, _ = jax.random.split(subkey)
        action, _ = inference_fn(state.obs, act_rng)
        step_fn(state, action)
        
        print("Simulation starting! Press Ctrl+C to stop and save.", flush=True)
        
        try:
            attempt_count = 0
            # We don't have viewer_obj.is_running(), so we loop indefinitely until Ctrl+C
            while True:
                attempt_count += 1
                if args.record_attempt is not None and attempt_count > args.record_attempt:
                    print(f"Recorded {args.record_attempt} attempts. Exiting...")
                    break

                # Reset at the start of each episode
                rng, rng_reset = jax.random.split(rng)
                state = reset_fn(rng_reset)

                # Custom Randomization (Copy of logic below)
                if args.randomize_domain or True: 
                     if 'PickCube' in env_name or 'PushCube' in env_name:
                         if args.randomize_domain: 
                            state = randomize_panda_pick_cube(state, rng)
                    
                     if 'HandOver' in env_name:
                         if args.randomize_domain: 
                            state = randomize_panda_handover(state, rng)
                     elif 'OpenCabinet' in env_name:
                         pass

                if args.randomize_joints:
                    if 'HandOver' in env_name:
                        state = randomize_joints_handover(state, rng)
                    else:
                        state = randomize_joints_global(state, rng)
                
                step_idx = 0
                episode_reward = 0
                
                # Episode Loop
                while True:
                    start_time = time.time()
                    
                    # Inference
                    rng, subkey = jax.random.split(rng)
                    act_rng, _ = jax.random.split(subkey)
                    action, _ = inference_fn(state.obs, act_rng)
                    
                    # Step Env
                    state = step_fn(state, action)
                    episode_reward += float(state.reward)
                    
                    # Sync Physics
                    q = np.array(state.data.qpos)
                    qd = np.array(state.data.qvel)

                    mj_data.qpos[:] = q
                    mj_data.qvel[:] = qd
                    
                    mujoco.mj_forward(mj_model, mj_data)

                    # Capture frame
                    renderer.update_scene(mj_data)
                    pixels = renderer.render()

                    # --- OVERLAY ATTEMPT NUMBER ---
                    img = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(img)
                    
                    # Attempt to load a nice font, fallback to default
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                    except IOError:
                        font = ImageFont.load_default()

                    text = f"Attempt: {attempt_count}/{args.record_attempt}"
                    # Right top corner
                    # Get text size to position it correctly
                    # textbbox available in newer PIL, textsize in older. 
                    # Let's try textbbox if available, else hardcode roughly or use textlength
                    try:
                        _, _, w, h = draw.textbbox((0, 0), text, font=font)
                    except AttributeError:
                        # Fallback for older PIL
                        w, h = draw.textsize(text, font=font)

                    # Position: Top Right with some padding
                    W, H = img.size
                    padding = 10
                    x = W - w - padding
                    y = padding
                    
                    # Draw text with some black outline for visibility
                    outline_color = (0, 0, 0)
                    text_color = (255, 255, 255)
                    
                    # Simple outline by drawing multiple times
                    for adj in [-1, 1]:
                        draw.text((x+adj, y), text, font=font, fill=outline_color)
                        draw.text((x, y+adj), text, font=font, fill=outline_color)
                    
                    draw.text((x, y), text, font=font, fill=text_color)
                    
                    pixels = np.array(img)
                    # ------------------------------

                    writer.append_data(pixels)
                    
                    # Timing
                    step_idx += 1
                    elapsed = time.time() - start_time
                    # Headless can run faster than real time if we want, but let's respect dt for consistency
                    # In headless, maybe we want it AS FAST AS POSSIBLE? 
                    # User said "simulasyon ne kadar sürmüşse" -> implies capturing the duration.
                    # But they also said "kasıyor" -> so headless should probably just process steps.
                    # If we sleep, it takes longer to record.
                    # Let's NOT sleep in headless mode to speed up recording! 
                    # But we must ensure FPS is correct in the saved GIF.
                    
                    if step_idx >= 200:
                        print(f"Episode done. Reward: {episode_reward:.2f}. Auto-resetting...")
                        break 
        except KeyboardInterrupt:
            print("\nCtrl+C pressed. Exiting simulation...", flush=True)
        finally:
            writer.close()
            print(f"Video saved to {video_path}", flush=True)
            
    else:
        # --- INTERACTIVE VIEWER MODE ---
        # Setup viewer
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer_obj:
            print("Viewer launched. Press Ctrl+C to exit.", flush=True)
            
            # --- JIT COMPILATION WARMUP ---
            print("\n--- INITIALIZING (JIT Compilation) ---")
            print("1. Compiling reset_fn...", flush=True)
            rng = jax.random.PRNGKey(args.seed)
            state = reset_fn(rng)
            print("   Done.", flush=True)

            # Sync immediately
            mj_data.qpos[:] = np.array(state.data.qpos)
            mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            viewer_obj.sync()
            
            print("2. Compiling step_fn & inference...", flush=True)
            rng, subkey = jax.random.split(rng)
            act_rng, _ = jax.random.split(subkey)
            
            # Trigger compilation
            action, _ = inference_fn(state.obs, act_rng)
            step_fn(state, action)
            print("   Done. Simulation starting!", flush=True)

            try:
                while viewer_obj.is_running():
                    # Reset at the start of each episode
                    rng, rng_reset = jax.random.split(rng)
                    state = reset_fn(rng_reset)

                    # --- CUSTOM RANDOMIZATION ---
                    if args.randomize_domain or True: # Enable by default for now if no specific flag logic is strict
                        if 'PickCube' in env_name or 'PushCube' in env_name:
                            if args.randomize_domain: # Use flag to confirm extra randomization
                                state = randomize_panda_pick_cube(state, rng)
                        if 'HandOver' in env_name:
                            if args.randomize_domain: # Use flag to confirm extra randomization
                                state = randomize_panda_handover(state, rng)
                        elif 'OpenCabinet' in env_name:
                            # Add logic if needed
                            pass

                    # 2. Global Joint Randomization
                    if args.randomize_joints:
                        state = randomize_joints_global(state, rng)
                    
                    # Reset counters
                    step_idx = 0
                    episode_reward = 0
                    
                    # Episode Loop
                    while True:
                        start_time = time.time()
                        
                        # Inference
                        rng, subkey = jax.random.split(rng)
                        act_rng, _ = jax.random.split(subkey)
                        action, _ = inference_fn(state.obs, act_rng)
                        
                        # Step Env
                        state = step_fn(state, action)
                        episode_reward += float(state.reward)
                        
                        # Sync Physics
                        q = np.array(state.data.qpos)
                        qd = np.array(state.data.qvel)

                        mj_data.qpos[:] = q
                        mj_data.qvel[:] = qd
                        
                        mujoco.mj_forward(mj_model, mj_data)
                        viewer_obj.sync()

                        # NO GIF CAPTURE HERE
                        
                        # Timing
                        step_idx += 1
                        elapsed = time.time() - start_time
                        sleep_time = env.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                        if step_idx >= 200: # Slightly longer episodes
                            print(f"Episode done. Reward: {episode_reward:.2f}. Auto-resetting...")
                            break 
                        
                        if not viewer_obj.is_running():
                            break
            except KeyboardInterrupt:
                print("\nCtrl+C pressed. Exiting simulation...", flush=True)        



if __name__ == "__main__":
    main()
