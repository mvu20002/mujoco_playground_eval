# MuJoCo Playground Mujoco Sim Demos

This repository contains training and **qualitative** evaluation pipelines for MuJoCo Playground environments using Brax PPO.
The focus is on manipulation tasks and policy generalization under distribution shifts.

MuJoCo Playground provides a collection of ready-to-use manipulation environments.
In this repository, the following environments are used:

- Panda-based environments
  - PandaOpenCabinet
  - PandaPickCube
  - PandaRobotiqPushCube

- Aloha-based environments
  - AlohaHandOver
  - AlohaSinglePegInsertion


## Evaluation Philosophy

Two different evaluation regimes are used:

### 1. Standard (In-Distribution) Evaluation
- The environment reset function is used as defined in the environment.
### 2. Out-of-Distribution (OOD) Evaluation
- Initial conditions are sampled outside the training randomization bounds.


## Simple Usage
You can list all available mujoco playground environments by running:
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

## Evaluations

### PandaPickCube

#### Environment Info

<details>
<summary><strong>Observation (Gözlem) Vektörü (Toplam: 39 boyut)</strong></summary>

<table>
  <tr><th>Bileşen</th><th>Açıklama</th><th>Boyut</th></tr>
  <tr><td>Robot eklem pozisyonları</td><td>data.qpos (7 kol eklemi + 2 parmak)</td><td>9</td></tr>
  <tr><td>Robot eklem hızları</td><td>data.qvel</td><td>9</td></tr>
  <tr><td>Gripper pozisyonu</td><td>gripper_pos (x, y, z)</td><td>3</td></tr>
  <tr><td>Gripper oryantasyonu</td><td>gripper_mat[3:] (matrisin son 6 elemanı)</td><td>6</td></tr>
  <tr><td>Kutu oryantasyonu</td><td>data.xmat[self._obj_body].ravel()[3:]</td><td>6</td></tr>
  <tr><td>Kutu ile gripper farkı</td><td>data.xpos[self._obj_body] - data.site_xpos[self._gripper_site]</td><td>3</td></tr>
  <tr><td>Hedef ile kutu farkı</td><td>info["target_pos"] - data.xpos[self._obj_body]</td><td>3</td></tr>
  <tr><td>Hedef ve kutu oryantasyon farkı</td><td>target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6]</td><td>6</td></tr>
  <tr><td>Kontrol ve pozisyon farkı</td><td><a href="http://vscodecontentref/8">data.ctrl - data.qpos[self._robot_qposadr[:-1]]</a></td><td>3</td></tr>
</table>
</details>

<details>
<summary><strong>Action (Aksiyon) Vektörü (Toplam: 9 boyut)</strong></summary>

- 7 kol eklemi (her biri için sürekli kontrol)
- 2 parmak eklemi (gripper aç/kapa)

</details>

#### Training Info

<table>
  <tr><th>Parametre</th><th>Değer</th></tr>
  <tr><td>num_timesteps</td><td>20,000,000</td></tr>
  <tr><td>num_evals</td><td>4</td></tr>
  <tr><td>unroll_length</td><td>10</td></tr>
  <tr><td>num_minibatches</td><td>32</td></tr>
  <tr><td>num_updates_per_batch</td><td>8</td></tr>
  <tr><td>discounting</td><td>0.97</td></tr>
  <tr><td>learning_rate</td><td>1e-3</td></tr>
  <tr><td>entropy_cost</td><td>2e-2</td></tr>
  <tr><td>num_envs</td><td>2048</td></tr>
  <tr><td>batch_size</td><td>512</td></tr>
  <tr><td>policy_hidden_layer_sizes</td><td>(32, 32, 32, 32)</td></tr>
</table>

#### In-Distribution

<p align="left">
  <video src="./results/PandaPickCubeOrientation/id.mp4" controls muted autoplay loop></video>
  <br>
  <em>Standard Environment</em>
</p>


#### Out-of-Distribution

<table align="left">
  <tr>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/arm.mp4" controls muted autoplay loop></video>
      <br>
      <em>Randomized Arm</em>
    </td>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/object.mp4" controls muted autoplay loop></video>
      <br>
      <em>Randomized Object</em>
    </td>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/both.mp4" controls muted autoplay loop></video>
      <br>
      <em>Randomized Both</em>
    </td>
  </tr>
</table>


### PandaOpenCabinet
not implemented
### PandaRobotiqPushCube
not implemented
### AlohaHandOver
not implemented
### AlohaSinglePegInsertion
not implemented


