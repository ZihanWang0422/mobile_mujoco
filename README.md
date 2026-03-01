# 🚁 Mobile MuJoCo — Drone MPC / MPPI Trajectory Tracking

Quadrotor (Crazyflie 2) trajectory tracking in MuJoCo physics simulation, implementing two Model Predictive Control approaches: **MPC** (CasADi nonlinear optimization) and **MPPI** (sampling-based Path Integral control).

---

## ✨ Features

- 🎯 **Dual MPC backends** — gradient-based IPOPT solver (MPC) and gradient-free sampling (MPPI)
- 🔄 **Cascade control architecture** — outer loop (50 Hz) for trajectory planning + inner AttitudePD loop (200 Hz)
- 🛤️ **Multiple trajectories** — circle, figure-8 (lemniscate), helix, with smooth warm-up ramp
- 🌀 **Spinning propeller animation** — visual-only prop rotation driven kinematically in MuJoCo
- 📷 **Camera tracking** — viewer follows the drone from above-side angle
- 🔴 **Flight trail visualization** — red line traces the drone's path in real time
- 📊 **Rich plotting** — 3D trajectory, per-axis tracking, error RMSE, control inputs
- ⚡ **Fast solves** — MPC ~2–5 ms, MPPI ~3–5 ms per outer-loop step

---

## 📁 Project Structure

```
mobile_mujoco/
├── models/
│   └── drone/
│       └── bitcraze_crazyflie_2/
│           ├── cf2.xml          # Crazyflie 2 MJCF model (with spinning props)
│           ├── scene.xml        # Simulation scene (ground + lighting)
│           └── assets/          # 3D mesh files
├── drone_mpc/
│   ├── drone_env.py             # MuJoCo environment wrapper
│   ├── mpc_controller.py        # MPC controller (CasADi + IPOPT)
│   ├── mppi_controller.py       # MPPI controller (sampling-based)
│   ├── inner_loop.py            # Attitude PD inner loop (200 Hz)
│   ├── trajectory.py            # Trajectory generators
│   └── visualization.py        # Plotting utilities
├── run_mpc.py                   # Run MPC tracking
├── run_mppi.py                  # Run MPPI tracking
├── run_compare.py               # MPC vs MPPI comparison
└── environment.yml              # Conda environment
```

---

## 🔧 Installation

**1. Create Conda environment**
```bash
conda create -n mobile_mujoco python=3.10 -y
conda activate mobile_mujoco
```

**2. Install dependencies**
```bash
pip install mujoco numpy scipy matplotlib casadi
```

Or use the provided environment file:
```bash
conda env create -f environment.yml
conda activate mobile_mujoco
```

**3. Verify**
```bash
python -c "import mujoco; import casadi; print('MuJoCo:', mujoco.__version__); print('All OK')"
```

---

## 🚀 Usage

### ▶️ MPC Trajectory Tracking

```bash
# Basic run (no viewer, generates result plots)
python run_mpc.py

# With real-time MuJoCo viewer
python run_mpc.py --render

# Custom parameters
python run_mpc.py --radius 1.5 --height 1.2 --omega 0.3 --duration 30 --render

# Figure-8 trajectory
python run_mpc.py --trajectory lemniscate --radius 2.0

# Save result plot
python run_mpc.py --save results/mpc_circle.png
```

### ▶️ MPPI Trajectory Tracking

```bash
# Basic run
python run_mppi.py

# With viewer
python run_mppi.py --render

# Tune MPPI parameters
python run_mppi.py --n-samples 512 --temperature 0.02 --horizon 40

# Save result
python run_mppi.py --save results/mppi_circle.png
```

### ⚖️ MPC vs MPPI Comparison

```bash
python run_compare.py
python run_compare.py --radius 1.0 --omega 0.5 --duration 30
python run_compare.py --save results/comparison.png
```

### 🎛️ CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--radius` | `1.0` | Trajectory radius (m) |
| `--height` | `1.0` | Flight altitude (m) |
| `--omega` | `0.5` | Angular speed (rad/s) |
| `--duration` | `20.0` | Simulation duration (s) |
| `--trajectory` | `circle` | `circle` \| `lemniscate` |
| `--render` | `False` | Enable MuJoCo real-time viewer |
| `--save` | `None` | Save plot to file path |
| `--horizon` | `25/30` | MPC/MPPI prediction horizon steps |
| `--n-samples` | `256` | MPPI sample count K |
| `--temperature` | `0.05` | MPPI temperature λ (lower = greedier) |

---

## 📝 License

MIT License
  - 支持 warm start（利用上一步解初始化）
  - 9D 简化状态 + RK4 动力学

### `drone_mpc/mppi_controller.py` — MPPI 控制器

- `MPPIController`: 采样路径积分控制
  - `compute_control(state, reference)`: MPPI 采样优化
  - 批量向量化前向仿真
  - 自适应控制序列平滑

### `drone_mpc/trajectory.py` — 轨迹生成

- `CircleTrajectory`: 水平圆形轨迹
- `LemniscateTrajectory`: 八字形轨迹
- `HelixTrajectory`: 螺旋上升轨迹


## 📝 License

MIT License
