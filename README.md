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

# Evaluations

## PandaPickCube

### Environment And Training Details

<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Gözlem (Observation)</th>
            <th>Açıklama</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>data.qpos</code></td>
            <td>Robotun eklem pozisyonları.</td>
        </tr>
        <tr>
            <td><code>data.qvel</code></td>
            <td>Robotun eklem hızları.</td>
        </tr>
        <tr>
            <td><code>gripper_pos</code></td>
            <td>Tutucunun (Gripper) Kartezyen pozisyonu.</td>
        </tr>
        <tr>
            <td><code>gripper_mat[3:]</code></td>
            <td>Tutucunun rotasyon matrisi (Kısmi/Düzleştirilmiş).</td>
        </tr>
        <tr>
            <td><code>obj_rot[3:]</code></td>
            <td>Küpün (Box) rotasyon matrisi (Kısmi).</td>
        </tr>
        <tr>
            <td><code>obj_pos - gripper_pos</code></td>
            <td>Küpün tutucuya göre bağıl konumu.</td>
        </tr>
        <tr>
            <td><code>target_pos - obj_pos</code></td>
            <td>Hedefin küpe göre bağıl konumu.</td>
        </tr>
        <tr>
            <td><code>target_mat - obj_mat</code></td>
            <td>Hedefin küpe göre bağıl rotasyon farkı.</td>
        </tr>
        <tr>
            <td><code>ctrl - qpos</code></td>
            <td>Mevcut kontrol sinyali ile eklem pozisyonu arasındaki fark (Action delta).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Eylem (Action)</th>
            <th>Açıklama</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>action</code></td>
            <td>Eklem kontrol sinyalleri (Delta Control). <code>action_scale</code> (0.04) ile çarpılarak mevcut <code>ctrl</code> değerine eklenir.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parametre</th>
            <th>Dağılım / Yöntem</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Küp Başlangıç Konumu (Box Pos)</td>
            <td>X ve Y eksenlerinde <code>[-0.2, 0.2]</code> aralığında rastgele konumlandırılır (Z sabit).</td>
        </tr>
        <tr>
            <td>Hedef Konumu (Target Pos)</td>
            <td>X ve Y'de <code>[-0.2, 0.2]</code>, Z ekseninde (yükseklik) <code>[0.2, 0.4]</code> aralığında rastgele belirlenir.</td>
        </tr>
        <tr>
            <td>Hedef Oryantasyonu (Target Quat)</td>
            <td>Sadece <code>sample_orientation=True</code> ise aktiftir. Rastgele bir eksen etrafında 0° ile 45° arasında rastgele döndürülür.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Ödül Bileşeni</th>
            <th>Ağırlık (Scale)</th>
            <th>Mantık</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>gripper_box</code></td>
            <td>4.0</td>
            <td>Tutucunun küpe yaklaşması (Mesafe azaldıkça ödül artar).</td>
        </tr>
        <tr>
            <td><code>box_target</code></td>
            <td>8.0</td>
            <td>Küpün hedefe yaklaşması. <strong>Not:</strong> Pozisyon hatası (%90) ve Rotasyon hatası (%10) ağırlıklı ortalaması alınır. Sadece <code>reached_box</code> (tutucu küpe ulaştıysa) durumunda aktiftir.</td>
        </tr>
        <tr>
            <td><code>no_floor_collision</code></td>
            <td>0.25</td>
            <td>Tutucu yere (floor) çarpmazsa 1, çarparsa 0.</td>
        </tr>
        <tr>
            <td><code>robot_target_qpos</code></td>
            <td>0.3</td>
            <td>Robot kolunun başlangıç duruşunu (init_q) korumaya çalışması (Regularization).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Durum</th>
            <th>Koşul</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Küpün pozisyonu mutlak değeri 1.0'ı aşarsa veya Z yüksekliği 0.0'ın altına düşerse (masadan düşerse).</td>
        </tr>
        <tr>
            <td>Numerical Error</td>
            <td>Simülasyondaki <code>qpos</code> veya <code>qvel</code> değerlerinde NaN (Not a Number) oluşursa.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
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
</details>


### In-Distribution Evaluation


<p align="left">
  <video src="./results/PandaPickCubeOrientation/id.mp4" controls muted autoplay loop></video>
  <br>
  <em>Standard Environment</em>
</p>


### Out-of-Distribution Evaluation

<table align="left">
  <tr>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/arm.mp4" controls muted autoplay loop></video>
      <br>
      <em>Arm Over Perturbation</em>
    </td>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/object.mp4" controls muted autoplay loop></video>
      <br>
      <em>Object Over Randomization</em>
    </td>
    <td align="left" width="33%">
      <video src="./results/PandaPickCubeOrientation/both.mp4" controls muted autoplay loop></video>
      <br>
      <em>Both</em>
    </td>
  </tr>
</table>


## PandaOpenCabinet

<details>
<summary><strong>Observation Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Gözlem (Observation)</th>
            <th>Açıklama</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>data.qpos</code></td>
            <td>Robotun eklem pozisyonları (Joint positions).</td>
        </tr>
        <tr>
            <td><code>data.qvel</code></td>
            <td>Robotun eklem hızları (Joint velocities).</td>
        </tr>
        <tr>
            <td><code>gripper_pos</code></td>
            <td>Tutucunun (Gripper) Kartezyen pozisyonu.</td>
        </tr>
        <tr>
            <td><code>gripper_mat[3:]</code></td>
            <td>Tutucunun rotasyon matrisi (Kısmi/Düzleştirilmiş).</td>
        </tr>
        <tr>
            <td><code>obj_rot[3:]</code></td>
            <td>Nesnenin (Handle/Cabinet) rotasyon matrisi (Kısmi).</td>
        </tr>
        <tr>
            <td><code>obj_pos - gripper_pos</code></td>
            <td>Nesnenin tutucuya göre bağıl konumu.</td>
        </tr>
        <tr>
            <td><code>target_pos - obj_pos</code></td>
            <td>Hedefin nesneye göre bağıl konumu.</td>
        </tr>
        <tr>
            <td><code>target_mat - obj_mat</code></td>
            <td>Hedefin nesneye göre bağıl rotasyon farkı.</td>
        </tr>
        <tr>
            <td><code>ctrl - qpos</code></td>
            <td>Mevcut kontrol sinyali ile eklem pozisyonu arasındaki fark (Action delta).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Action Vector</strong></summary>
<table>
    <thead>
        <tr>
            <th>Eylem (Action)</th>
            <th>Açıklama</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>action</code></td>
            <td>Eklem kontrol sinyalleri (Delta Control). <code>action_scale</code> (0.04) ile çarpılarak mevcut <code>ctrl</code> değerine eklenir.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Randomizations</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parametre</th>
            <th>Dağılım / Yöntem</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Hedef Pozisyonu (Target Pos)</td>
            <td>Başlangıç [0.3, 0.0, 0.5] üzerine X ekseninde <code>[-0.1, 0.1]</code> aralığında uniform gürültü eklenir.</td>
        </tr>
        <tr>
            <td>Robot Kolu Başlangıç Açıları (Init Qpos)</td>
            <td>Kol eklemlerine (ilk 7 eklem) <code>[-30°, +30°]</code> (yaklaşık) aralığında rastgele bozulma (perturbation) uygulanır.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Reward Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Ödül Bileşeni</th>
            <th>Ağırlık (Scale)</th>
            <th>Mantık</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>gripper_box</code></td>
            <td>4.0</td>
            <td>Tutucunun kutuya/nesneye olan mesafesi (yaklaştıkça artar: <code>1 - tanh</code>).</td>
        </tr>
        <tr>
            <td><code>box_target</code></td>
            <td>8.0</td>
            <td>Kutunun hedefe olan mesafesi. Yalnızca <code>reached_box</code> (tutucu nesneye ulaştıysa) 1.0 olduğunda aktif olur.</td>
        </tr>
        <tr>
            <td><code>no_barrier_collision</code></td>
            <td>0.25</td>
            <td>Bariyerle çarpışma yoksa ödül verilir (Çarpışma varsa 0, yoksa 1).</td>
        </tr>
        <tr>
            <td><code>robot_target_qpos</code></td>
            <td>0.3</td>
            <td>Robot kolunun başlangıç pozisyonundan (init_q) çok uzaklaşmamasını teşvik eder (Regularization).</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Done Mechanism</strong></summary>
<table>
    <thead>
        <tr>
            <th>Durum</th>
            <th>Koşul</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Out of Bounds</td>
            <td>Nesne (Box) pozisyonu mutlak değeri 1.0'ı aşarsa veya Z yüksekliği 0.0'ın altına düşerse.</td>
        </tr>
        <tr>
            <td>Numerical Error</td>
            <td>Simülasyondaki <code>qpos</code> veya <code>qvel</code> değerlerinde NaN (Not a Number) oluşursa.</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary><strong>Training Details</strong></summary>
<table>
    <thead>
        <tr>
            <th>Parametre</th>
            <th>Değer</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>num_timesteps</td>
            <td>40_000_000</td>
        </tr>
        <tr>
            <td>num_evals</td>
            <td>4</td>
        </tr>
        <tr>
            <td>unroll_length</td>
            <td>10</td>
        </tr>
        <tr>
            <td>num_minibatches</td>
            <td>32</td>
        </tr>
        <tr>
            <td>num_updates_per_batch</td>
            <td>8</td>
        </tr>
        <tr>
            <td>discounting</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>learning_rate</td>
            <td>1e-3</td>
        </tr>
        <tr>
            <td>entropy_cost</td>
            <td>2e-2</td>
        </tr>
        <tr>
            <td>num_envs</td>
            <td>2048</td>
        </tr>
        <tr>
            <td>batch_size</td>
            <td>512</td>
        </tr>
        <tr>
            <td>policy_hidden_layer_sizes</td>
            <td>(32, 32, 32, 32)</td>
        </tr>
    </tbody>
</table>
</details>    


### In-Distribution Evaluation

<p align="left">
  <video src="./results/PandaOpenCabinet/id.mp4" controls muted autoplay loop></video>
  <br>
  <em>Standart Environment</em>
</p>

### Out-of-Distribution Evaluation

<p align="left">
  <video src="./results/PandaOpenCabinet/arm.mp4" controls muted autoplay loop></video>
  <br>
  <em>Arm Over Perturbation</em>
</p>


## AlohaHandOver
not written yet
## AlohaSinglePegInsertion
not written yet