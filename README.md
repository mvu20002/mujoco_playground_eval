# MuJoCo Playground Mujoco Sim Demos

This repository contains training and **qualitative** evaluation pipelines for MuJoCo Playground environments using Brax PPO.
The focus is on manipulation tasks and policy generalization under distribution shifts.

MuJoCo Playground provides a collection of ready-to-use manipulation environments.
In this repository, the following environments are used:

1) PandaOpenCabinet
2) PandaPickCube
3) AlohaHandOver
4) AlohaSinglePegInsertion


### Evaluation Philosophy

Two different evaluation regimes are used:

#### 1. Standard (In-Distribution) Evaluation
- The environment reset function is used as defined in the environment.

#### 2. Out-of-Distribution (OOD) Evaluation
- Initial conditions are sampled outside the training randomization bounds.


### Simple Usage
Once you install [mujoco playground](https://github.com/google-deepmind/mujoco_playground), you can list all available mujoco playground environments by running:
```bash
python -c "from mujoco_playground import registry; print(registry.manipulation.ALL_ENVS)"
```
out:
```text
('AlohaHandOver', 'AlohaSinglePegInsertion', 'PandaPickCube', 'PandaPickCubeOrientation', 'PandaPickCubeCartesian', 'PandaOpenCabinet', 'PandaRobotiqPushCube', 'LeapCubeReorient', 'LeapCubeRotateZAxis', 'AeroCubeRotateZAxis')
```

To train a policy, run:
```bash
python train.py --env_name <env_name>
```

To evaluate a policy, run:
```bash
python test_in_mujoco.py --env_name <env_name> --policy_path <policy_path>
```

---

# 1) PandaPickCube

### Environment And Training Details

<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Observation</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Positions (qpos)</td>
            <td>Current angles of the robot joints.</td>
        </tr>
        <tr>
            <td>Joint Velocities (qvel)</td>
            <td>Current angular velocities of the robot joints.</td>
        </tr>
        <tr>
            <td>Gripper Position</td>
            <td>Cartesian position (XYZ) of the gripper site.</td>
        </tr>
        <tr>
            <td>Gripper Orientation</td>
            <td>Flattened rotation matrix of the gripper (indices 3 onwards).</td>
        </tr>
        <tr>
            <td>Box Orientation</td>
            <td>Flattened rotation matrix of the box body (indices 3 onwards).</td>
        </tr>
        <tr>
            <td>Box Relative to Gripper</td>
            <td>Vector difference: Box Position - Gripper Position.</td>
        </tr>
        <tr>
            <td>Target Relative to Box</td>
            <td>Vector difference: Target Position - Box Position.</td>
        </tr>
        <tr>
            <td>Orientation Error</td>
            <td>Difference between Target rotation matrix and Box rotation matrix (first 6 elements).</td>
        </tr>
        <tr>
            <td>Control Error</td>
            <td>Difference between current Control input and Joint Positions (for the robot arm).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Action</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Control Delta</td>
            <td>Continuous control signals added to current joint positions (scaled by <code>0.04</code>).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Distribution / Method</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Box Initial Position</td>
            <td>Uniform random within +/- 0.2m (X, Y) from initial object position. Z is fixed at 0.0.</td>
        </tr>
        <tr>
            <td>Target Position</td>
            <td>Uniform random within +/- 0.2m (X, Y) and +0.2m to +0.4m (Z) from initial object position.</td>
        </tr>
        <tr>
            <td>Target Orientation</td>
            <td>Random axis with random angle (up to 45 deg) if <code>sample_orientation=True</code>. Otherwise fixed [1,0,0,0].</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Reward Component</th>
            <th>Weight (Scale)</th>
            <th>Logic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>gripper_box</code></td>
            <td>4.0</td>
            <td>Proximity of the gripper to the box (tanh-scaled).</td>
        </tr>
        <tr>
            <td><code>box_target</code></td>
            <td>8.0</td>
            <td>Combined position and rotation error between box and target. <strong>Only active if box is reached</strong>.</td>
        </tr>
        <tr>
            <td><code>no_floor_collision</code></td>
            <td>0.25</td>
            <td>Penalty (0 reward) if the hand/fingers touch the floor sensors.</td>
        </tr>
        <tr>
            <td><code>robot_target_qpos</code></td>
            <td>0.3</td>
            <td>Encourages the arm to stay close to its initial configuration (regularization).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Status</th>
            <th>Condition</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Box position X or Y > 1.0, or Box Z < 0.0.</td>
        </tr>
        <tr>
            <td>Simulation Error</td>
            <td>NaN values detected in <code>qpos</code> or <code>qvel</code>.</td>
        </tr>
        <tr>
            <td>Timeout</td>
            <td>Implied by standard environment limits (though explicit timeout logic isn't in the `step` function shown, it usually defaults to `episode_length`).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Num Timesteps</td>
            <td>20,000,000</td>
        </tr>
        <tr>
            <td>Episode Length</td>
            <td>150</td>
        </tr>
        <tr>
            <td>Control DT / Sim DT</td>
            <td>0.02 / 0.005</td>
        </tr>
        <tr>
            <td>Num Environments</td>
            <td>2048</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Num Minibatches</td>
            <td>32</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>1e-3</td>
        </tr>
        <tr>
            <td>Discounting (Gamma)</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>Policy Hidden Layers</td>
            <td>(32, 32, 32, 32)</td>
        </tr>
        <tr>
            <td>Action Scale</td>
            <td>0.04</td>
        </tr>
    </tbody>
</table>
</details>




### Evaluation

#### Environment Reset
<video src="https://github.com/user-attachments/assets/f6a32fe2-49e1-431d-b0ec-3d64705da0d3"></video>

#### Arm Initial Configuration Over Randomization
<video src="https://github.com/user-attachments/assets/dd75296a-7df1-4cc5-a18c-440fcf4ceaaf"></video>

#### Box Initial Configuration Over Randomization
<video src="https://github.com/user-attachments/assets/4c1d8dd0-ee28-4e18-8e71-21eea2832085"></video>

#### Both Arm and Box Initial Configuration Over Randomization
<video src="https://github.com/user-attachments/assets/4ce2672b-4fd6-4a70-ad41-7b80a441d294"></video>



# 2) PandaOpenCabinet

<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Observation</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Positions (qpos)</td>
            <td>Current angles of the robot joints.</td>
        </tr>
        <tr>
            <td>Joint Velocities (qvel)</td>
            <td>Current angular velocities of the robot joints.</td>
        </tr>
        <tr>
            <td>Gripper Position</td>
            <td>Cartesian position (XYZ) of the gripper site.</td>
        </tr>
        <tr>
            <td>Gripper Orientation</td>
            <td>Flattened rotation matrix of the gripper (indices 3 onwards).</td>
        </tr>
        <tr>
            <td>Handle Orientation</td>
            <td>Flattened rotation matrix of the object/handle (indices 3 onwards).</td>
        </tr>
        <tr>
            <td>Handle Relative to Gripper</td>
            <td>Vector difference: Handle Position - Gripper Position.</td>
        </tr>
        <tr>
            <td>Target Relative to Handle</td>
            <td>Vector difference: Target Position - Handle Position.</td>
        </tr>
        <tr>
            <td>Orientation Error</td>
            <td>Difference between Target rotation matrix and Handle rotation matrix (first 6 elements).</td>
        </tr>
        <tr>
            <td>Control Error</td>
            <td>Difference between current Control input and Joint Positions.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Action</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Control Delta</td>
            <td>Continuous control signals added to current joint positions (scaled by <code>0.04</code>).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Distribution / Method</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Target Position (X-axis)</td>
            <td>Base [0.3, 0.0, 0.5]. X-coordinate perturbed by Uniform [-0.1, 0.1].</td>
        </tr>
        <tr>
            <td>Initial Arm Configuration</td>
            <td>Initial joint angles perturbed by random noise (approx +/- 30 degrees range per joint).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Reward Component</th>
            <th>Weight (Scale)</th>
            <th>Logic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>gripper_box</code></td>
            <td>4.0</td>
            <td>Proximity of the gripper to the cabinet handle (tanh-scaled).</td>
        </tr>
        <tr>
            <td><code>box_target</code></td>
            <td>8.0</td>
            <td>Proximity of the handle to the target position. <strong>Only active if gripper is close to handle</strong> (grasped).</td>
        </tr>
        <tr>
            <td><code>no_barrier_collision</code></td>
            <td>0.25</td>
            <td>Penalty (0 reward) if the hand/fingers touch the barrier geom.</td>
        </tr>
        <tr>
            <td><code>robot_target_qpos</code></td>
            <td>0.3</td>
            <td>Encourages the arm to stay close to its initial configuration (regularization).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Status</th>
            <th>Condition</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Object position X or Y > 1.0, or Object Z < 0.0.</td>
        </tr>
        <tr>
            <td>Simulation Error</td>
            <td>NaN values detected in <code>qpos</code> or <code>qvel</code>.</td>
        </tr>
        <tr>
            <td>Timeout</td>
            <td>Steps >= <code>episode_length</code> (150).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Num Timesteps</td>
            <td>40,000,000</td>
        </tr>
        <tr>
            <td>Episode Length</td>
            <td>150</td>
        </tr>
        <tr>
            <td>Control DT / Sim DT</td>
            <td>0.02 / 0.005</td>
        </tr>
        <tr>
            <td>Num Environments</td>
            <td>2048</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Num Minibatches</td>
            <td>32</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>1e-3</td>
        </tr>
        <tr>
            <td>Discounting (Gamma)</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>Policy Hidden Layers</td>
            <td>(32, 32, 32, 32)</td>
        </tr>
        <tr>
            <td>Action Scale</td>
            <td>0.04</td>
        </tr>
    </tbody>
</table>
</details>


### Evaluation

#### Environment Reset
<video src="https://github.com/user-attachments/assets/03d6d3c5-d6be-4984-bf45-73cad2be1d8b"></video>

#### Arm Initial Configuration Over Randomization
<video src="https://github.com/user-attachments/assets/87217a83-c6dc-492b-8e7d-590f545c1334"></video>

---
# 3) AlohaHandOver
<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Observation</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Positions (qpos)</td>
            <td>Current angles of the robot joints.</td>
        </tr>
        <tr>
            <td>Joint Velocities (qvel)</td>
            <td>Current angular velocities of the robot joints.</td>
        </tr>
        <tr>
            <td>Finger Grasp Margin</td>
            <td>Finger positions relative to the box width (<code>finger_qposadr - box_width</code>).</td>
        </tr>
        <tr>
            <td>Box Sites</td>
            <td>Positions of the box's top and bottom sites.</td>
        </tr>
        <tr>
            <td>Left Gripper Pose</td>
            <td>Position and orientation (partial rotation matrix) of the left gripper.</td>
        </tr>
        <tr>
            <td>Right Gripper Pose</td>
            <td>Position and orientation (partial rotation matrix) of the right gripper.</td>
        </tr>
        <tr>
            <td>Box Orientation</td>
            <td>Orientation (partial rotation matrix) of the box body.</td>
        </tr>
        <tr>
            <td>Target Relative Position</td>
            <td>Vector difference between the box position and the random target position.</td>
        </tr>
        <tr>
            <td>Time</td>
            <td>Normalized episode time (current step / episode length).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Action</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Control Delta</td>
            <td>Continuous control signals added to current joint positions (scaled by <code>0.015</code>) for the ALOHA bimanual robot arms.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Distribution / Method</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Box Initial Position (XY)</td>
            <td>Uniform: X in range [-0.05, 0.05], Y in range [-0.1, 0.1].</td>
        </tr>
        <tr>
            <td>Target Position</td>
            <td>Base [0.20, 0.0, 0.25] + Uniform noise [-0.15, 0.15]. X-coordinate clipped to min 0.15.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Reward Component</th>
            <th>Weight (Scale)</th>
            <th>Logic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>gripper_box</code></td>
            <td>1.0</td>
            <td>Proximity of Left Gripper to Box Top and Right Gripper to Box Bottom.</td>
        </tr>
        <tr>
            <td><code>box_handover</code></td>
            <td>4.0</td>
            <td>Distance of the box (or left gripper) to the handover point [0.0, 0.0, 0.24].</td>
        </tr>
        <tr>
            <td><code>handover_target</code></td>
            <td>8.0</td>
            <td>Distance of the box to the random target position (biased towards Right Gripper holding it).</td>
        </tr>
        <tr>
            <td><code>no_table_collision</code></td>
            <td>0.3</td>
            <td>Reward for avoiding collisions between grippers/box and the table.</td>
        </tr>
        <tr>
            <td>Dropping Penalty</td>
            <td>N/A (Fixed -0.1)</td>
            <td>Penalty applied if the box is dropped (height < 0.05) after being successfully picked up.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Status</th>
            <th>Condition</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Box position X or Y > 1.0, or Box Z < 0.0 (fell off table).</td>
        </tr>
        <tr>
            <td>Simulation Error</td>
            <td>NaN values detected in <code>qpos</code> or <code>qvel</code>.</td>
        </tr>
        <tr>
            <td>Dropped</td>
            <td>Box Z < 0.05 AND <code>episode_picked</code> flag is True.</td>
        </tr>
        <tr>
            <td>Timeout</td>
            <td>Current steps >= <code>episode_length</code> (250).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Num Timesteps</td>
            <td>100,000,000</td>
        </tr>
        <tr>
            <td>Episode Length</td>
            <td>250 (5 seconds)</td>
        </tr>
        <tr>
            <td>Control DT / Sim DT</td>
            <td>0.02 / 0.005</td>
        </tr>
        <tr>
            <td>Num Environments</td>
            <td>2048</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Num Minibatches</td>
            <td>32</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>1e-3</td>
        </tr>
        <tr>
            <td>Discounting (Gamma)</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>Policy Hidden Layers</td>
            <td>(256, 256, 256)</td>
        </tr>
        <tr>
            <td>Action Scale</td>
            <td>0.015</td>
        </tr>
    </tbody>
</table>
</details> 

### Evaluation

Note: Training was done for 88.473.600 timesteps instead of 100.000.000 timesteps.

#### Environment Reset
<video src="https://github.com/user-attachments/assets/03d6d3c5-d6be-4984-bf45-73cad2be1d8b"></video>

#### Box Initial Configuration Over Randomization
<video src="https://github.com/user-attachments/assets/03d6d3c5-d6be-4984-bf45-73cad2be1d8b"></video>

# 4) AlohaSinglePegInsertion
<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Observation</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Positions (qpos)</td>
            <td>Current angles of the robot joints.</td>
        </tr>
        <tr>
            <td>Joint Velocities (qvel)</td>
            <td>Current angular velocities of the robot joints.</td>
        </tr>
        <tr>
            <td>Left Gripper Position</td>
            <td>Cartesian position (XYZ) of the left gripper site.</td>
        </tr>
        <tr>
            <td>Socket Body Position</td>
            <td>Cartesian position (XYZ) of the socket object.</td>
        </tr>
        <tr>
            <td>Right Gripper Position</td>
            <td>Cartesian position (XYZ) of the right gripper site.</td>
        </tr>
        <tr>
            <td>Peg Body Position</td>
            <td>Cartesian position (XYZ) of the peg object.</td>
        </tr>
        <tr>
            <td>Socket Entrance Position</td>
            <td>Position of the specific site marking the socket hole entrance.</td>
        </tr>
        <tr>
            <td>Peg Tip Position</td>
            <td>Position of the peg's insertion tip (peg_end2).</td>
        </tr>
        <tr>
            <td>Socket Orientation Z</td>
            <td>Z-axis vector of the socket's rotation matrix (indicates uprightness).</td>
        </tr>
        <tr>
            <td>Peg Orientation Z</td>
            <td>Z-axis vector of the peg's rotation matrix (indicates uprightness).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Action</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Joint Control Delta</td>
            <td>Continuous control signals added to current joint positions (scaled by <code>0.005</code>) for the bimanual ALOHA robot.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Distribution / Method</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Peg Initial Position (XY)</td>
            <td>Uniform random noise [-0.1, 0.1] added to initial peg joint coordinates.</td>
        </tr>
        <tr>
            <td>Socket Initial Position (XY)</td>
            <td>Uniform random noise [-0.1, 0.1] added to initial socket joint coordinates.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Reward Component</th>
            <th>Weight (Scale)</th>
            <th>Logic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>left_reward</code> / <code>right_reward</code></td>
            <td>1.0</td>
            <td>Reward for Left Gripper being close to Socket, and Right Gripper being close to Peg.</td>
        </tr>
        <tr>
            <td><code>socket_entrance_reward</code></td>
            <td>4.0</td>
            <td>Reward for lifting the socket entrance to the goal height (approx 0.15m).</td>
        </tr>
        <tr>
            <td><code>peg_end2_reward</code></td>
            <td>4.0</td>
            <td>Reward for lifting the peg tip to the goal height (approx 0.15m).</td>
        </tr>
        <tr>
            <td><code>socket_z_up</code> / <code>peg_z_up</code></td>
            <td>0.5</td>
            <td>Reward for keeping both objects upright (Z-axis alignment), multiplied by their lift progress.</td>
        </tr>
        <tr>
            <td><code>peg_insertion_reward</code></td>
            <td>8.0</td>
            <td>Reward for minimizing distance between Peg Tip and Socket Rear. <strong>Only active if aligned</strong> (peg is within 5mm of the insertion line).</td>
        </tr>
        <tr>
            <td><code>left/right_target_qpos</code></td>
            <td>0.3</td>
            <td>Regularization to keep arms near home pose (gated by gripping success).</td>
        </tr>
        <tr>
            <td><code>no_table_collision</code></td>
            <td>0.3</td>
            <td>Reward for avoiding collisions with the table.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Status</th>
            <th>Condition</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Socket or Peg position > 1.0 (in any axis).</td>
        </tr>
        <tr>
            <td>Simulation Error</td>
            <td>NaN values detected in <code>qpos</code> or <code>qvel</code>.</td>
        </tr>
        <tr>
            <td>Timeout</td>
            <td>Steps >= <code>episode_length</code> (1000).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Num Timesteps</td>
            <td>150,000,000</td>
        </tr>
        <tr>
            <td>Episode Length</td>
            <td>1000</td>
        </tr>
        <tr>
            <td>Control DT / Sim DT</td>
            <td>0.0025 / 0.0025</td>
        </tr>
        <tr>
            <td>Num Environments</td>
            <td>1024</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Num Minibatches</td>
            <td>32</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>3e-4</td>
        </tr>
        <tr>
            <td>Discounting (Gamma)</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>Policy Hidden Layers</td>
            <td>(256, 256, 256, 256)</td>
        </tr>
        <tr>
            <td>Action Scale</td>
            <td>0.005</td>
        </tr>
        <tr>
            <td>Entropy Cost</td>
            <td>1e-2</td>
        </tr>
    </tbody>
</table>
</details>

### Evaluation
Note: This model unfortunatelly couldnt trained in local machine due high training time.