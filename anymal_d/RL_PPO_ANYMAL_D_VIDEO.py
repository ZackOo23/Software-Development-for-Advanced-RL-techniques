"""
Render evaluation videos for a trained ANYmal D PPO policy (MuJoCo).

Companion to RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py — it uses the same
MultivariateNormal `Agent`, so it loads checkpoints produced by that trainer.

Ported from the ANYmal C version. Changes: C -> D model path / save dir, the
hardcoded checkpoint path replaced by a CLI arg with an auto-pick fallback, and
an optional Hugging Face Hub download of a checkpoint. This script does not need
an experiment tracker; the original wandb video-log line was already disabled.

Usage:
    python RL_PPO_ANYMAL_D_VIDEO.py                       # auto-pick best local checkpoint
    python RL_PPO_ANYMAL_D_VIDEO.py --policy path/to/x_policy.pt
    python RL_PPO_ANYMAL_D_VIDEO.py --num-videos 5 --std-init 0.85
    python RL_PPO_ANYMAL_D_VIDEO.py --hf-repo user/anymal-d-ppo --hf-file x_policy.pt
"""

import os
import re
import glob
import argparse
import numpy as np
import mujoco

from typing import List
import mediapy as media
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

MUJOCO_STEPS = 5


# --- Paths -------------------------------------------------------------------
def find_project_file(rel_path):
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
os.makedirs(SAVE_DIR, exist_ok=True)


class Agent(nn.Module):
    def __init__(self, obs_len, act_len):
        super(Agent, self).__init__()

        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, act_len)
        )
        self.critic = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, state):
        out = self.mlp(state)
        action_scores = self.actor(out)
        state_value = self.critic(out)
        return action_scores, state_value

    def compute_action(self, state, action_std):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self(state)
        probs = torch.tanh(probs)

        action_var = torch.full((self.act_len,), action_std * action_std)
        cov_mat = torch.diag(action_var).unsqueeze(dim=0)

        m = torch.distributions.multivariate_normal.MultivariateNormal(probs, cov_mat)

        action = m.sample()

        action_clamped = torch.tanh(action)

        action_clamped[0][0] = action_clamped[0][0] * 0.6 - 0.1
        action_clamped[0][3] = action_clamped[0][3] * 0.6 + 0.1
        action_clamped[0][6] = action_clamped[0][6] * 0.6 - 0.1
        action_clamped[0][9] = action_clamped[0][9] * 0.6 + 0.1

        return (
            action_clamped.detach().numpy(),
            m.log_prob(action_clamped).detach().numpy(),
            state_value.detach(),
        )


class Env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.FRAMERATE = 60  # Hz
        self.DURATION = 8  # seconds
        self.TIMESTEP = 0.002  # 0.002 by default
        self.done = False
        self.model.opt.timestep = self.TIMESTEP
        # Make a new camera, move it to a closer distance.
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 5
        self.frames = []

    def reset(self):
        # Simulate and save data
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl = np.zeros(12)
        state = np.array(self.data.qpos.copy())
        state = np.append(state, self.data.qvel.copy())
        self.frames.clear()
        return state

    def step(self, action, render=False):
        self.done = False
        reward = 0
        self.data.ctrl = action
        for i in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)
            reward = reward + self.data.qvel[0] + (self.data.qpos[2] - 0.5)
            if render and (len(self.frames) < self.data.time * self.FRAMERATE):
                self.camera.lookat = self.data.body("LH_SHANK").subtree_com
                self.renderer.update_scene(self.data, self.camera)
                pixels = self.renderer.render()
                self.frames.append(pixels.copy())

        state = np.array(self.data.qpos.copy())
        state = np.append(state, self.data.qvel.copy())
        if self.data.time > self.DURATION:
            self.done = True
        if self.data.qpos[2] < 0.3:
            self.done = True
            reward = reward - 100
        return state, reward, self.done

    def close(self, episode, reward):
        path = os.path.join(SAVE_DIR, f"video_{episode}_reward_{reward}.mp4")
        media.write_video(path, self.frames, fps=self.FRAMERATE)
        return path


def test(action_std, env, policy, num_video, render=False):
    state, ep_reward, done = env.reset(), 0, False
    counter = 0
    reward_list = []
    cumulative_reward_list = []
    time_list = []
    while not done:
        action, _, _ = policy.compute_action(state, action_std)
        state, reward, done = env.step(action, render=True)
        reward_list.append(reward)
        time_list.append(counter * 0.002 * MUJOCO_STEPS)
        ep_reward += reward
        cumulative_reward_list.append(ep_reward)
        counter = counter + 1

    env.close(num_video + 1, ep_reward)
    # (Experiment-tracker video logging is intentionally off in this script.)

    # Plotting episode reward
    plt.figure()
    plt.plot(time_list, reward_list)
    plt.plot(time_list, cumulative_reward_list)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.title("Episode instant and cumulative Reward")
    plt.savefig(os.path.join(SAVE_DIR, f"video_reward_{num_video + 1}.png"))
    plt.close()

    return ep_reward


def pick_latest_checkpoint(save_dir=SAVE_DIR):
    candidates = glob.glob(os.path.join(save_dir, "*_policy.pt")) + glob.glob(
        os.path.join(save_dir, "*olicy*.pt")
    )
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No policy checkpoints found in {save_dir}. Train first or pass --policy."
        )

    def score(path):
        m = re.search(r"Reward[-_]([-\d.]+)", os.path.basename(path))
        reward = float(m.group(1)) if m else float("-inf")
        return (reward, os.path.getmtime(path))

    return max(candidates, key=score)


def maybe_download_from_hub(repo_id, filename):
    """Optionally fetch a checkpoint from a Hugging Face Hub model repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id, filename=filename, token=os.environ.get("HF_TOKEN")
    )
    print(f"Downloaded {filename} from {repo_id} -> {path}")
    return path


def make_video(num_videos=10, std_init=0.85820, policy_path=None):
    env = Env()

    # Fix random seed (for reproducibility)
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if policy_path is None:
        policy_path = pick_latest_checkpoint()
    print(f"Loading policy from: {policy_path}")
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    policy.eval()

    action_std = std_init

    for i_video in range(num_videos):
        ep_reward = test(action_std, env, policy, i_video)
        print(f"Video #{i_video + 1} reward: {ep_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render ANYmal D evaluation videos")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument(
        "--std-init",
        type=float,
        default=0.85820,
        help="action std used at evaluation time",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="path to a *_policy.pt checkpoint; defaults to the best in pretrained_models/anymal_d",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="optional Hugging Face Hub model repo to download a checkpoint from",
    )
    parser.add_argument(
        "--hf-file",
        type=str,
        default=None,
        help="filename within --hf-repo to download",
    )
    args = parser.parse_args()

    policy_path = args.policy
    if args.hf_repo and args.hf_file:
        policy_path = maybe_download_from_hub(args.hf_repo, args.hf_file)

    make_video(
        num_videos=args.num_videos, std_init=args.std_init, policy_path=policy_path
    )
