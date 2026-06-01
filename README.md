# PPO for ANYmal D (MuJoCo) — Hugging Face

This is the ANYmal **D** version of the ANYmal C PPO project. The three training
scripts are direct ports of the ANYmal C set, with two kinds of change only:

1. **ANYmal D.** They load the ANYmal D model (`anybotics_anymal_d/scene.xml`) and
   save to `pretrained_models/anymal_d/`. The robot's kinematics are identical
   between C and D (12 leg joints, 19-dim `qpos`, 18-dim `qvel`), so the network
   sizes, action mapping, and reward logic are unchanged. The standing pose and
   height target are read from the model's `home` keyframe, so they track the D
   XML automatically.
2. **Hugging Face.** All experiment tracking now uses the Hugging Face
   stack instead of Weights & Biases (see "What changed" below).

## Layout

```
anymal_d_ppo/
├── anymal_d/                                  # the scripts
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py          # MultivariateNormal trainer + sweep
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py# Normal/delta trainer + sweep + render
│   ├── RL_PPO_ANYMAL_D_VIDEO.py                   # render videos from a checkpoint
│   └── experiment_metrics.py                      # CSV/JSONL metrics logger
├── anybotics_anymal_d/                        # the MuJoCo model (scene.xml, assets, …)
├── pretrained_models/anymal_d/                # checkpoints + videos land here
├── runs/<run_name>/                           # structured experiment logs
├── requirements.txt
└── README.md
```

The scripts find `anybotics_anymal_d/scene.xml` by walking up from their own
location, so you can run them from anywhere in the project.

## Install

```bash
pip install -r requirements.txt
# headless rendering (Linux): apt-get install libosmesa6 && pip install PyOpenGL
```

## Run

```bash
# 1) Older trainer (MultivariateNormal policy, raw-action control)
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py            # single run
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py --sweep    # random-search sweep

# 2) Polished trainer (recommended for the assignment rubric: structured CSV/JSONL metrics)
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --run-name ppo_baseline_01
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --live      # live viewer
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py sweep --sweep-count 30
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py render --num-videos 5

# 3) Make evaluation videos from the best checkpoint
python anymal_d/RL_PPO_ANYMAL_D_VIDEO.py --num-videos 5
# headless: prefix any of the above with  MUJOCO_GL=osmesa
```

The video script pairs with trainer **1** (same MultivariateNormal policy class).
Trainer **2** renders its own videos via its `render` mode.



## Experiment tracking and metrics logging

This repository records metrics in two places:

1. **Trackio dashboard** for interactive visualization. Start a run normally and then open:

```bash
trackio show --project AIDL-PPO-ANYMAL_D
```

2. **Local structured logs** for reproducibility. The polished trainer always writes:

```text
runs/<run_name>/config.json
runs/<run_name>/metrics.csv
runs/<run_name>/metrics.jsonl
```

The CSV and JSONL logs contain one row/object per training or evaluation episode. The recorded fields include episode return, running return, episode length, PPO losses, entropy, PPO ratio, action standard deviation, success rate, fall rate, forward distance, path length, waypoint completion rate, and control smoothness.

Run an experiment with a readable name:

```bash
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --run-name ppo_baseline_01
```

Then inspect the structured metrics:

```bash
head runs/ppo_baseline_01/metrics.csv
python -m json.tool runs/ppo_baseline_01/config.json
```

Metric definitions are saved inside `config.json`. In this PPO baseline, `success` means the robot does not fall and reaches `success_distance_m` of forward progress. `waypoint_completion_rate` is computed from the default forward waypoints `[0.25, 0.5, 1.0, 1.5, 2.0]` meters so the PPO baseline can later be compared against the imitation-learning waypoint policy. `control_smoothness` is the mean squared change between consecutive bounded actions; lower values mean smoother control.

## Optional environment variables

Everything runs **fully offline** without these. Set them to enable the cloud bits:

```bash
export HF_MODEL_REPO="your-username/anymal-d-ppo"   # enable checkpoint upload
export HF_TOKEN="hf_..."                            # write token (or: hf auth login)
export HF_PRIVATE=1                                 # 1=private repo (default), 0=public
export TRACKIO_SPACE_ID="your-username/anymal-d-dash"  # host the dashboard on a Space
```

View the local dashboard any time with:

```bash
trackio show --project AIDL-PPO-ANYMAL_D
```

## Notes / caveats

- The two trainers are intentionally different policies (the C set shipped both):
  trainer 1 uses a `MultivariateNormal` with an annealed action std; trainer 2 uses
  a `Normal` with a learnable `log_std` and acts as a small delta around the
  nominal pose. Their checkpoints are **not** interchangeable.
- ANYmal D's link masses/geometry differ from C (the thigh is notably heavier),
  so reward magnitudes and the episodes needed to "solve" will differ — expect to
  re-tune rather than reuse C's numbers. The HAA action remap (`0.6*x ± 0.1`) sits
  inside D's hip limits (~[-0.785, 0.611]).
- If `loss.backward()` segfaults on a CPU-only box via a GPU torch build, remove
  the stray `triton` package (`pip uninstall triton`); torch then falls back cleanly.
```
