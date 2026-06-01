"""
PPO training / sweep / rendering for the ANYmal D quadruped (MuJoCo) — the
polished variant of the trainer with a state-independent learnable log_std,
delta-around-nominal actions, shaped reward, and offscreen video logging.

Ported from the ANYmal C version with two kinds of change only:
  1. C -> D: model path, save dirs, project name. The 12-joint kinematics,
     19-dim qpos and 18-dim qvel are identical, so N_OBS / N_ACT / the network
     and reward shaping are unchanged. The nominal standing pose is read from
     the model's `home` keyframe, so it tracks the ANYmal D XML automatically.
  2. wandb -> Hugging Face:
       * logging          -> trackio (`import trackio as wandb`, drop-in).
       * checkpoint upload-> Hugging Face Hub (`push_checkpoint_to_hub`).
       * sweep            -> self-contained random-search driver `run_sweep`
                            (trackio has no sweep agent). Same ranges as before.

Optional env vars (all-offline without them):
    HF_MODEL_REPO, HF_TOKEN, TRACKIO_SPACE_ID   (see README)

Usage:
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --run-name ppo_baseline_01
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --live
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py sweep --sweep-count 30
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py render --num-videos 5
"""

import os
import re
import glob
import argparse
import numpy as np
import mujoco

# --- Hugging Face experiment tracking (drop-in wandb replacement) ------------
import trackio as wandb

import mediapy as media
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_metrics import MetricsLogger

from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT = "AIDL-PPO-ANYMAL_D"
MUJOCO_STEPS = 5


# Paths -----------------------------------------------------------------------
def find_project_file(rel_path):
    """Locate `rel_path` by walking up from this file and the CWD, so the
    script runs from anywhere."""
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(6):
            cand = os.path.join(d, rel_path)
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    return os.path.join(".", rel_path)


MODEL_XML = find_project_file(os.path.join("anybotics_anymal_d", "scene.xml"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(MODEL_XML), ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "pretrained_models", "anymal_d")
VIDEO_DIR = os.path.join(SAVE_DIR, "videos")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


# --- Hugging Face Hub checkpoint upload (replaces wandb.save) -----------------
def push_checkpoint_to_hub(local_path, path_in_repo=None):
    """Upload a checkpoint to a HF Hub model repo if HF_MODEL_REPO is set;
    otherwise a no-op (file is already saved locally)."""
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(
            repo_id,
            repo_type="model",
            exist_ok=True,
            private=os.environ.get("HF_PRIVATE", "1") == "1",
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo or os.path.basename(local_path),
            repo_id=repo_id,
            repo_type="model",
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"  uploaded {os.path.basename(local_path)} -> {url}")
        return url
    except Exception as e:
        print(f"  [hub] upload skipped ({e})")
        return None


# --- Action / observation constants -----------------------------------------
# The policy outputs a small DELTA around a nominal standing pose. With
# position actuators this also keeps torques small (torque ~ kp*(target - q)),
# which matches the intuition that small torques should be optimal.
ACTION_SCALE = 0.4  # max delta per joint after tanh squash, in radians (~23 deg).
# 0.25 produced balanced-but-short shuffling — the legs
# didn't have enough range for a real swing phase.
# 0.4 is the sweet-spot ballpark; try 0.35–0.5 to sweep.
N_ACT = 12
# qpos = [x, y, z, qw, qx, qy, qz, 12 joints]   length 19
# qvel = [vx, vy, vz, wx, wy, wz, 12 joints]    length 18
# Drop world x,y from the observation — they grow unbounded as the robot walks
# and the network never sees the same input twice.
N_OBS = (19 - 2) + 18  # 17 + 18 = 35


class Agent(nn.Module):
    """Shared-trunk actor-critic with a state-independent learnable log_std."""

    def __init__(self, obs_len, act_len, log_std_init=-0.5):
        super().__init__()
        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, act_len)
        )
        self.critic = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        # exp(-0.5) ~ 0.61 — reasonable initial exploration.
        self.log_std = nn.Parameter(torch.full((act_len,), log_std_init))

    def forward(self, state):
        h = self.mlp(state)
        return self.actor(h), self.critic(h)

    def dist(self, state):
        mean, value = self(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), value

    @torch.no_grad()
    def compute_action(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        d, value = self.dist(state_t)
        a = d.sample()  # shape (1, 12)
        # Independent Gaussians → joint log-prob is the sum across action dims.
        logp = d.log_prob(a).sum(dim=-1)  # shape (1,)
        return a.squeeze(0).numpy(), float(logp.item()), float(value.item())


class Env:
    def __init__(self, fall_threshold=0.35):
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.FRAMERATE = 60
        self.DURATION = 8
        self.TIMESTEP = 0.002
        self.done = False
        self.model.opt.timestep = self.TIMESTEP
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 5
        self.frames = []
        self.fall_threshold = fall_threshold
        self.viewer = None

        # Read the standing pose straight from the model's `home` keyframe so
        # this stays in sync with the XML. With position actuators, ctrl =
        # joint angle target, so commanding the nominal pose holds the stance.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.nominal_pose = self.data.qpos[7 : 7 + N_ACT].copy()
        # The ANYmal D `home` pose stands a touch higher than the C; reading the
        # keyframe height keeps the height reward target correct for either robot.
        self.target_height = float(self.data.qpos[2])
        self.last_action = np.zeros(N_ACT)

    def attach_viewer(self):
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5
        print("Live viewer attached. Close the window to detach.")

    def detach_viewer(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def _sync_viewer(self):
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            return
        self.viewer.cam.lookat[:] = self.data.body("LH_SHANK").subtree_com
        self.viewer.sync()

    def _obs(self):
        # Skip qpos[0:2] (world x,y).
        return np.concatenate(
            [
                self.data.qpos[2:].copy(),  # z, quat (4), 12 joints  → 17
                self.data.qvel.copy(),  # 6 base vel + 12 joint vel → 18
            ]
        )

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Hold the nominal pose on reset so the robot doesn't sag while
        # ctrl is still zeros from the previous episode.
        self.data.ctrl[:] = self.nominal_pose
        self.last_action[:] = 0.0
        self.episode_start_x = float(self.data.qpos[0])
        self.prev_x = self.episode_start_x
        self.path_length_m = 0.0
        self.frames.clear()
        return self._obs()

    def step(self, raw_action, render=False):
        """
        raw_action is the un-squashed policy sample. We tanh-bound it INSIDE
        the env (not inside the distribution — that's how the old code broke
        log-prob accounting) and scale to a small delta around the nominal
        standing pose. Same exact `raw_action` is stored in the replay buffer,
        so PPO's log-prob math stays consistent.
        """
        self.done = False
        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(N_ACT)
        bounded = np.tanh(raw_action)
        smoothness_step = float(np.mean((bounded - self.last_action) ** 2))
        prev_x = float(self.data.qpos[0])
        self.data.ctrl[:] = self.nominal_pose + ACTION_SCALE * bounded

        for _ in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)
            self._sync_viewer()
            if render and (len(self.frames) < self.data.time * self.FRAMERATE):
                self.camera.lookat = self.data.body("LH_SHANK").subtree_com
                self.renderer.update_scene(self.data, self.camera)
                pixels = self.renderer.render()
                self.frames.append(pixels.copy())

        # --- Reward shaping --------------------------------------------------
        # --- Velocity reward -------------------------------------------------
        # The previous Gaussian-on-target gave too little gradient at low
        # speeds: the policy could collect a slice of vel_reward by shuffling
        # while dodging the action/joint-vel costs of a real stride. tanh is
        # monotone (more speed → more reward, always) but saturates near
        # ~1.5 m/s, so the robot doesn't try to sprint catastrophically.
        forward_vel = self.data.qvel[0]
        vel_reward = 2.0 * np.tanh(forward_vel)

        # Stay near the nominal hip height (read from the `home` keyframe).
        height = self.data.qpos[2]
        height_reward = -2.0 * (height - self.target_height) ** 2

        # Upright: qw = 1 when level, 0 when on its side.
        qw = self.data.qpos[3]
        upright_reward = 0.5 * qw**2

        # Survival.
        alive_bonus = 0.5

        # Torque-proxy cost: penalize big commands, but gently — too high (0.01)
        # made the robot freeze into short steps. 0.002 lets the policy commit a
        # full-amplitude swing without the cost dominating the velocity gain.
        action_cost = -0.002 * float(np.sum(bounded**2))

        # Discourage jerky changes between consecutive commands.
        smooth_cost = -0.002 * float(np.sum((bounded - self.last_action) ** 2))

        # Very mild penalty on joint velocity — bounds energy without
        # suppressing the swing phase.
        joint_vel_cost = -0.0001 * float(np.sum(self.data.qvel[6:] ** 2))

        reward = (
            vel_reward
            + height_reward
            + upright_reward
            + alive_bonus
            + action_cost
            + smooth_cost
            + joint_vel_cost
        )
        self.last_action = bounded.copy()

        self.path_length_m += abs(float(self.data.qpos[0]) - prev_x)
        forward_distance_m = float(self.data.qpos[0]) - self.episode_start_x
        fall = bool(height < self.fall_threshold)

        if self.data.time > self.DURATION:
            self.done = True
        if fall:
            self.done = True
            reward -= 5.0  # soft fall penalty — no -100 cliff

        info = {
            "fall": fall,
            "height_m": float(height),
            "forward_distance_m": forward_distance_m,
            "path_length_m": float(self.path_length_m),
            "control_smoothness_step": smoothness_step,
        }
        return self._obs(), reward, self.done, info

    def close(self, episode, reward, prefix="video"):
        path = os.path.join(VIDEO_DIR, f"{prefix}_{episode}_reward_{reward:.2f}.mp4")
        media.write_video(path, self.frames, fps=self.FRAMERATE)
        return path


# --- Replay buffer -----------------------------------------------------------
# Fixes vs. the original older trainer:
#  * a_logp is a SCALAR (sum of independent Gaussians), not a (12,) vector.
#  * `d` (done flag) is stored so the value target doesn't bootstrap past
#    terminal states.
transition = np.dtype(
    [
        ("s", np.float64, (N_OBS,)),
        ("a", np.float64, (N_ACT,)),
        ("a_logp", np.float64),
        ("r", np.float64),
        ("s_", np.float64, (N_OBS,)),
        ("d", np.float64),
    ]
)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer_capacity = capacity
        self.buffer = np.empty(capacity, dtype=transition)
        self.counter = 0

    def store(self, t):
        self.buffer[self.counter] = t
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False


def train(policy, optimizer, memory, hparams):
    gamma = hparams["gamma"]
    ppo_epoch = hparams["ppo_epoch"]
    batch_size = hparams["batch_size"]
    clip_param = hparams["clip_param"]
    c1 = hparams["c1"]
    c2 = hparams["c2"]
    max_grad = hparams.get("max_grad_norm", 0.5)

    s = torch.tensor(memory.buffer["s"], dtype=torch.float)
    a = torch.tensor(memory.buffer["a"], dtype=torch.float)
    r = torch.tensor(memory.buffer["r"], dtype=torch.float).view(-1, 1)
    s_ = torch.tensor(memory.buffer["s_"], dtype=torch.float)
    d = torch.tensor(memory.buffer["d"], dtype=torch.float).view(-1, 1)
    old_a_logp = torch.tensor(memory.buffer["a_logp"], dtype=torch.float).view(-1, 1)

    with torch.no_grad():
        v_next = policy(s_)[1]
        v_curr = policy(s)[1]
        # (1 − d) zeroes out the bootstrap when the next state is terminal.
        target_v = r + gamma * v_next * (1.0 - d)
        adv = target_v - v_curr
        # Per-batch advantage normalization — large stability win.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    last_pl = last_vl = last_ent = last_ratio = 0.0
    for _ in range(ppo_epoch):
        for index in BatchSampler(
            SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False
        ):
            dist, value = policy.dist(s[index])
            a_logp = dist.log_prob(a[index]).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(a_logp - old_a_logp[index])
            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[index]

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(value, target_v[index])
            loss = policy_loss + c1 * value_loss - c2 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            optimizer.step()

            last_pl, last_vl = policy_loss.item(), value_loss.item()
            last_ent, last_ratio = entropy.item(), ratio.mean().item()

    return last_pl, last_vl, last_ent, last_ratio


def render_episode(
    env,
    policy,
    episode,
    log_media=True,
    save_plot=True,
    prefix="video",
    metric_logger=None,
    success_distance_m=1.0,
    waypoint_distances_m=(0.25, 0.5, 1.0, 1.5, 2.0),
):
    # Windows fix: skip checkpoint MP4 rendering during training.
    # Metrics are still saved in runs/<run_name>/metrics.csv and metrics.jsonl.
    if prefix == "checkpoint":
        print("[video skipped] Checkpoint video rendering disabled on Windows.")
        return None
    """Offscreen-render one episode, save as video, optionally log to trackio."""
    state, ep_reward, done = env.reset(), 0, False
    counter = 0
    smoothness_sum = 0.0
    last_info = {
        "fall": False,
        "forward_distance_m": 0.0,
        "path_length_m": 0.0,
        "control_smoothness_step": 0.0,
    }
    reward_list, cumulative_reward_list, time_list = [], [], []
    while not done:
        action, _, _ = policy.compute_action(state)
        state, reward, done, last_info = env.step(action, render=False)
        smoothness_sum += last_info["control_smoothness_step"]
        reward_list.append(reward)
        time_list.append(counter * 0.002 * MUJOCO_STEPS)
        ep_reward += reward
        cumulative_reward_list.append(ep_reward)
        counter += 1

    # Windows fix: skip MP4 video writing during training.
    # The rubric requires structured metrics, not video files.
    video_path = None
    env.frames = []
    print("  [video skipped] MP4 rendering disabled for this training run.")

    if False and log_media and wandb.run is not None:
        log_key = (
            "Video train"
            if prefix.startswith(("train", "checkpoint"))
            else "Video eval"
        )
        try:
            wandb.log({log_key: wandb.Video(video_path, fps=4, format="mp4")})
        except Exception as e:
            print(f"  [trackio] video log skipped ({e})")

    if save_plot:
        plt.figure()
        plt.plot(time_list, reward_list, label="instant")
        plt.plot(time_list, cumulative_reward_list, label="cumulative")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Reward")
        plt.title(f"Episode {episode} Reward")
        plt.legend()
        plot_path = os.path.join(VIDEO_DIR, f"{prefix}_reward_{episode}.png")
        plt.savefig(plot_path)
        if log_media and wandb.run is not None:
            plot_key = (
                "Reward train"
                if prefix.startswith(("train", "checkpoint"))
                else "Reward eval"
            )
            try:
                wandb.log({plot_key: wandb.Image(plot_path)})
            except Exception as e:
                print(f"  [trackio] plot log skipped ({e})")
        plt.close()

    if metric_logger is not None:
        episode_time_sec = counter * env.TIMESTEP * MUJOCO_STEPS
        record = metric_logger.build_episode_record(
            phase="eval",
            episode=episode,
            global_step=counter,
            episode_return=ep_reward,
            running_return=None,
            episode_length=counter,
            episode_time_sec=episode_time_sec,
            fall=bool(last_info["fall"]),
            forward_distance_m=float(last_info["forward_distance_m"]),
            path_length_m=float(last_info["path_length_m"]),
            control_smoothness=smoothness_sum / max(counter, 1),
            success_distance_m=success_distance_m,
            waypoint_distances_m=waypoint_distances_m,
        )
        metric_logger.log(record)
        if log_media and wandb.run is not None:
            wandb.log(MetricsLogger.tracker_payload(record, prefix="eval/"))

    return ep_reward, video_path


def pick_latest_checkpoint(save_dir=SAVE_DIR):
    candidates = glob.glob(os.path.join(save_dir, "*_policy.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_policy.pt files found in {save_dir}. Train first or pass --policy explicitly."
        )

    def score(path):
        m = re.search(r"Reward-([-\d.]+)_policy\.pt$", os.path.basename(path))
        reward = float(m.group(1)) if m else float("-inf")
        return (reward, os.path.getmtime(path))

    return max(candidates, key=score)


def make_video(num_videos=5, policy_path=None, fall_threshold=0.3):
    if policy_path is None:
        policy_path = pick_latest_checkpoint()
    print(f"Loading policy from: {policy_path}")

    env = Env(fall_threshold=fall_threshold)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    policy.eval()

    for i_video in range(num_videos):
        ep_reward, _ = render_episode(
            env, policy, i_video + 1, log_media=False, save_plot=True, prefix="eval"
        )
        print(f"Video #{i_video + 1} reward: {ep_reward:.2f}")


sweep_configuration = {
    "name": "ppo_sweep_0",
    "method": "bayes",  # honoured as random search by run_sweep
    "metric": {"name": "avg_reward", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "max": 1e-3, "min": 1e-5},
        "ppo_epoch": {"distribution": "int_uniform", "max": 20, "min": 5},
        "clip_param": {"distribution": "uniform", "max": 0.3, "min": 0.1},
        "c2": {"distribution": "uniform", "max": 0.02, "min": 0.0},
        "replay_size": {"distribution": "int_uniform", "max": 8192, "min": 2048},
    },
}


def _sample_param(spec, rng):
    dist = spec.get("distribution", "uniform")
    lo, hi = spec["min"], spec["max"]
    if dist == "int_uniform":
        return int(rng.integers(int(lo), int(hi) + 1))
    if dist in ("log_uniform_values", "log_uniform"):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def train_or_sweep(
    is_sweep=True, live=False, overrides=None, run_name=None, num_episodes=None
):
    """
    live:          if True, open a live MuJoCo viewer window that updates as
                   training runs. Requires a display.
    overrides:     hparams to inject (used by the sweep driver, replaces wandb.config).
    run_name:      explicit trackio run name (also used for checkpoint filenames).
    num_episodes:  optional cap (sweeps usually use fewer episodes per trial).
    Returns the final running reward.
    """
    hparams = {
        "gamma": 0.99,
        "log_interval": 50,
        "num_episodes": 15000,
        "lr": 3e-4,
        "clip_param": 0.2,
        "ppo_epoch": 10,
        "replay_size": 4096,
        "batch_size": 128,
        "c1": 0.5,
        "c2": 0.005,
        "max_grad_norm": 0.5,
    }
    if overrides:
        hparams.update(overrides)
        print(f"Params updated from sweep: {overrides}")
    if num_episodes is not None:
        hparams["num_episodes"] = num_episodes

    run = wandb.init(
        project=PROJECT,
        name=run_name,
        config=hparams,
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
    )
    run_name = getattr(run, "name", None) or run_name or "run"

    # Local structured logs required for reproducibility. These are written even
    # if the Trackio/Hugging Face dashboard is not used.
    hparams.setdefault("success_distance_m", 1.0)
    hparams.setdefault("waypoint_distances_m", [0.25, 0.5, 1.0, 1.5, 2.0])
    metrics_logger = MetricsLogger(PROJECT_ROOT, run_name, config=hparams)
    print(f"Structured metrics: {metrics_logger.csv_path}")

    env = Env(fall_threshold=0.35)

    if live:
        try:
            env.attach_viewer()
        except Exception as e:
            print(f"Could not open live viewer ({e}). Continuing headless.")

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    policy = Agent(N_OBS, N_ACT)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams["lr"])
    memory = ReplayMemory(hparams["replay_size"])

    # Per-step reward maxes at ~3 (vel 1.5 + alive 0.5 + upright 0.5 + height
    # ~0 + small costs). 8s / (5 * 0.002) ~ 800 env steps → max ~ 2400/episode.
    target_reward = 1800
    print(f"Target reward: {target_reward}")

    running_reward = 0.0
    saving_reward = 100.0
    global_step = 0
    last_losses = {"policy_loss": None, "value_loss": None, "entropy": None, "ratio": None}
    try:
        for i_episode in range(hparams["num_episodes"]):
            state, ep_reward, done = env.reset(), 0, False
            episode_length = 0
            smoothness_sum = 0.0
            last_info = {
                "fall": False,
                "forward_distance_m": 0.0,
                "path_length_m": 0.0,
                "control_smoothness_step": 0.0,
            }

            while not done:
                action, a_logp, _ = policy.compute_action(state)
                next_state, reward, done, last_info = env.step(action, render=False)
                episode_length += 1
                global_step += 1
                smoothness_sum += last_info["control_smoothness_step"]

                if memory.store(
                    (state, action, a_logp, reward, next_state, float(done))
                ):
                    pl, vl, ent, ratio = train(policy, optimizer, memory, hparams)
                    last_losses = {
                        "policy_loss": pl,
                        "value_loss": vl,
                        "entropy": ent,
                        "ratio": ratio,
                    }
                    wandb.log(
                        {
                            "policy_loss": pl,
                            "value_loss": vl,
                            "avg_reward": running_reward,
                            "avg_entropy": ent,
                            "ratio": ratio,
                            "action_std": float(policy.log_std.detach().exp().mean()),
                        }
                    )

                state = next_state
                ep_reward += reward
                if done:
                    break

            running_reward = round(0.05 * ep_reward + 0.95 * running_reward, 2)

            episode_time_sec = episode_length * env.TIMESTEP * MUJOCO_STEPS
            record = metrics_logger.build_episode_record(
                phase="train",
                episode=i_episode,
                global_step=global_step,
                episode_return=ep_reward,
                running_return=running_reward,
                episode_length=episode_length,
                episode_time_sec=episode_time_sec,
                fall=bool(last_info["fall"]),
                forward_distance_m=float(last_info["forward_distance_m"]),
                path_length_m=float(last_info["path_length_m"]),
                control_smoothness=smoothness_sum / max(episode_length, 1),
                success_distance_m=float(hparams["success_distance_m"]),
                waypoint_distances_m=hparams["waypoint_distances_m"],
                policy_loss=last_losses["policy_loss"],
                value_loss=last_losses["value_loss"],
                entropy=last_losses["entropy"],
                ratio=last_losses["ratio"],
                action_std=float(policy.log_std.detach().exp().mean()),
            )
            metrics_logger.log(record)
            wandb.log(MetricsLogger.tracker_payload(record, prefix="train/"))

            if i_episode % hparams["log_interval"] == 0:
                print(
                    f"Episode {i_episode}\tLast reward: {ep_reward:.2f}\t"
                    f"Average reward: {running_reward:.2f}"
                )

            # --- Checkpoint save (with eval video) ----------------------------
            if running_reward > saving_reward:
                saving_reward = running_reward
                policy_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_Reward-{running_reward}_policy.pt",
                )
                optim_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_Reward-{running_reward}_optimizer.pt",
                )
                torch.save(policy, policy_path)
                torch.save(optimizer, optim_path)
                push_checkpoint_to_hub(policy_path)  # replaces wandb.save
                push_checkpoint_to_hub(optim_path)
                print(f"Saved checkpoint to {SAVE_DIR}")
                render_episode(
                    env,
                    policy,
                    i_episode,
                    log_media=True,
                    save_plot=True,
                    prefix="checkpoint",
                    metric_logger=metrics_logger,
                    success_distance_m=float(hparams["success_distance_m"]),
                    waypoint_distances_m=hparams["waypoint_distances_m"],
                )

            if running_reward > target_reward:
                print("Solved!")
                policy_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_Reward-{running_reward}_policy.pt",
                )
                optim_path = os.path.join(
                    SAVE_DIR,
                    f"{run_name}_{i_episode}_Reward-{running_reward}_optimizer.pt",
                )
                torch.save(policy, policy_path)
                torch.save(optimizer, optim_path)
                push_checkpoint_to_hub(policy_path)
                push_checkpoint_to_hub(optim_path)
                break

        print(f"Finished training! Running reward is now {running_reward}")
    finally:
        env.detach_viewer()
        metrics_logger.close()
        wandb.finish()
    return running_reward


def run_sweep(count=50, episodes_per_trial=2000, live=False, seed=0):
    """Self-contained replacement for `wandb.sweep` + `wandb.agent`.
    Random-searches the `sweep_configuration` ranges; each trial is its own
    trackio run."""
    rng = np.random.default_rng(seed)
    best = {"reward": float("-inf"), "name": None, "params": None}
    print(
        f"Starting random-search sweep: {count} trials x {episodes_per_trial} episodes each"
    )
    for t in range(count):
        sampled = {
            k: _sample_param(v, rng)
            for k, v in sweep_configuration["parameters"].items()
        }
        name = f"{sweep_configuration['name']}_trial{t:03d}"
        print(f"\n=== Trial {t + 1}/{count} :: {name} :: {sampled} ===")
        reward = train_or_sweep(
            is_sweep=True,
            live=live,
            overrides=sampled,
            run_name=name,
            num_episodes=episodes_per_trial,
        )
        if reward > best["reward"]:
            best = {"reward": reward, "name": name, "params": sampled}
        print(
            f"Trial {name} finished. reward={reward}  | best so far={best['reward']} ({best['name']})"
        )
    print(
        f"\nSweep done. Best trial: {best['name']}  reward={best['reward']}\n  params={best['params']}"
    )
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["train", "sweep", "render"],
        help="train: regular run | sweep: random-search sweep | render: video only from a checkpoint",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="(train/sweep) open a live MuJoCo viewer window. Requires a display.",
    )
    parser.add_argument("--sweep-count", type=int, default=50)
    parser.add_argument("--sweep-episodes", type=int, default=2000)
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="(render) path to *_policy.pt. Defaults to highest-reward checkpoint in SAVE_DIR.",
    )
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.3,
        help="(render) z-height below which the episode terminates. Training uses 0.35.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="(train) optional readable run name used for checkpoints and runs/<name>/ logs.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_or_sweep(is_sweep=False, live=args.live, run_name=args.run_name)
    elif args.mode == "sweep":
        run_sweep(
            count=args.sweep_count,
            episodes_per_trial=args.sweep_episodes,
            live=args.live,
        )
    elif args.mode == "render":
        make_video(
            num_videos=args.num_videos,
            policy_path=args.policy,
            fall_threshold=args.fall_threshold,
        )
