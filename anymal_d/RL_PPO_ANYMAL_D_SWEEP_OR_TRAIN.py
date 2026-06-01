"""
PPO training / hyperparameter sweep for the ANYmal D quadruped (MuJoCo).

Ported from the ANYmal C version. Only two kinds of change were made:
  1. C -> D: model path, save dirs, project name. The kinematics (12 leg
     joints, 19-dim qpos, 18-dim qvel) are identical between C and D, so the
     network sizes, action mapping and reward are unchanged.
  2. wandb -> Hugging Face:
       * metric / video / plot logging  -> trackio  (`import trackio as wandb`,
         a drop-in wandb-compatible API that stores runs locally and can sync a
         free dashboard to a Hugging Face Space).
       * checkpoint upload (`wandb.save`) -> Hugging Face Hub model repo via
         `huggingface_hub` (see `push_checkpoint_to_hub`).
       * hyperparameter sweep (`wandb.sweep`/`wandb.agent`, which trackio does
         not provide) -> a small self-contained random-search driver,
         `run_sweep`, that honours the same `sweep_configuration` ranges and
         logs every trial as its own trackio run.

Optional environment variables (everything works fully offline without them):
    HF_MODEL_REPO     e.g. "your-username/anymal-d-ppo"  -> enables Hub upload
    HF_TOKEN          your Hugging Face write token       (or use `hf auth login`)
    TRACKIO_SPACE_ID  e.g. "your-username/anymal-d-dash"  -> host the dashboard

Usage:
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py            # single training run
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py --sweep    # random-search sweep
    python RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py --sweep --sweep-count 30 --sweep-episodes 2000
"""

import os
import glob
import argparse
import numpy as np
import mujoco

# --- Hugging Face experiment tracking (drop-in wandb replacement) ------------
import trackio as wandb

from typing import List
import mediapy as media
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT = "AIDL-PPO-ANYMAL_D"
MUJOCO_STEPS = 5


# --- Paths -------------------------------------------------------------------
def find_project_file(rel_path):
    """Locate `rel_path` (e.g. 'anybotics_anymal_d/scene.xml') by walking up
    from this file and from the current working directory. Keeps the scripts
    runnable regardless of where you launch them from."""
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(6):
            cand = os.path.join(d, rel_path)
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    # Fall back to the conventional relative path.
    return os.path.join(".", rel_path)


MODEL_XML = find_project_file(os.path.join("anybotics_anymal_d", "scene.xml"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(MODEL_XML), ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "pretrained_models", "anymal_d")
os.makedirs(SAVE_DIR, exist_ok=True)


# --- Hugging Face Hub checkpoint upload (replaces wandb.save) -----------------
def push_checkpoint_to_hub(local_path, path_in_repo=None):
    """Upload a checkpoint to a Hugging Face Hub model repo if HF_MODEL_REPO is
    set; otherwise this is a no-op (the file is already saved locally)."""
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

        # The four HAA (hip abduction/adduction) joints get remapped to their
        # asymmetric travel. ANYmal D HAA limits are ~[-0.785, 0.611] (front)
        # and mirrored on the hind legs; 0.6*x +/- 0.1 stays well inside them.
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
        if self.data.qpos[2] < 0.45:
            self.done = True
            reward = reward - 100
        return state, reward, self.done

    def close(self, episode, reward, prefix="video"):
        path = os.path.join(VIDEO_DIR, f"{prefix}_{episode}_reward_{reward:.2f}.mp4")
        try:
            media.write_video(path, self.frames, fps=self.FRAMERATE)
            return path
        except Exception as e:
            print(f"[video skipped] Could not write video: {e}")
            return None


# qpos (19) + qvel (18) = 37 observation dims, 12 actions. Same as ANYmal C.
transition = np.dtype(
    [
        ("s", np.float64, (37,)),
        ("a", np.float64, (12,)),
        ("a_logp", np.float64, (12,)),
        ("r", np.float64),
        ("s_", np.float64, (37,)),
    ]
)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer_capacity = capacity
        self.buffer = np.empty(capacity, dtype=transition)
        self.counter = 0

    # Stores a transition and returns True or False depending on whether the buffer is full or not
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False


def train(policy, optimizer, memory, hparams, action_std):

    gamma = hparams["gamma"]
    ppo_epoch = hparams["ppo_epoch"]
    batch_size = hparams["batch_size"]
    clip_param = hparams["clip_param"]
    c1 = hparams["c1"]
    c2 = hparams["c2"]

    s = torch.tensor(memory.buffer["s"], dtype=torch.float)
    a = torch.tensor(memory.buffer["a"], dtype=torch.float)
    r = torch.tensor(memory.buffer["r"], dtype=torch.float).view(-1, 1)
    s_ = torch.tensor(memory.buffer["s_"], dtype=torch.float)

    old_a_logp = torch.tensor(memory.buffer["a_logp"], dtype=torch.float).view(-1, 1)
    action_var = torch.full((12,), action_std * action_std)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv = target_v - policy(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(
            SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False
        ):
            probs, _ = policy(s[index])
            dist = MultivariateNormal(probs, cov_mat)
            entropy = dist.entropy()

            a_logp = dist.log_prob(a[index]).unsqueeze(dim=1)

            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]

            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[index]

            policy_loss = torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(policy(s[index])[1], target_v[index])
            entropy = entropy.mean()

            loss = -policy_loss + c1 * value_loss - c2 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return -policy_loss.item(), value_loss.item(), entropy.item(), ratio.mean().item()


def test(action_std, env, policy, episode, render=False):
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

    video_path = env.close(episode, ep_reward)
    # trackio.Video has the same signature as wandb.Video.
    try:
        wandb.log({"Video eval": wandb.Video(video_path, fps=4, format="mp4")})
    except Exception as e:
        print(f"  [trackio] video log skipped ({e})")

    # Plot episode reward. trackio.log takes media objects (Image), not the raw
    # pyplot module, so we save the figure and log it as an Image.
    plt.figure()
    plt.plot(time_list, reward_list, label="instant")
    plt.plot(time_list, cumulative_reward_list, label="cumulative")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.title("Episode instant and cumulative Reward")
    plt.legend()
    plot_path = os.path.join(SAVE_DIR, f"reward_eval_{episode}.png")
    plt.savefig(plot_path)
    plt.close()
    try:
        wandb.log({"Reward eval": wandb.Image(plot_path)})
    except Exception as e:
        print(f"  [trackio] plot log skipped ({e})")

    return ep_reward


# Same parameter ranges as the original wandb sweep. Used by `run_sweep` below.
sweep_configuration = {
    "name": "ppo_sweep_0",
    "method": "bayes",  # honoured as random search by run_sweep
    "metric": {"name": "avg_reward", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "max": 0.0001, "min": 0.00001},
        "ppo_epoch": {"distribution": "int_uniform", "max": 60, "min": 40},
        "c2": {"distribution": "uniform", "max": 0.01, "min": 0.001},
        "replay_size": {"distribution": "int_uniform", "max": 10000, "min": 6000},
        "std_init": {"distribution": "uniform", "max": 1.1, "min": 1.0},
        "std_min": {"distribution": "uniform", "max": 0.8, "min": 0.7},
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


def train_or_sweep(is_sweep=True, overrides=None, run_name=None, num_episodes=None):
    """
    overrides:     dict of hparams to inject (used by the sweep driver in place
                   of the old `wandb.config`).
    run_name:      explicit trackio run name (also used for checkpoint filenames).
    num_episodes:  optional cap (sweeps usually use fewer episodes per trial).
    Returns the final running reward (so the sweep can rank trials).
    """
    hparams = {
        "gamma": 0.99,
        "log_interval": 50,
        "num_episodes": 15000,
        "lr": 1e-5,
        "clip_param": 0.1,
        "ppo_epoch": 48,
        "replay_size": 6400,
        "batch_size": 128,
        "c1": 1.0,
        "c2": 0.001,
        "std_init": 1.0,
        "std_min": 0.6,
    }
    if overrides:
        hparams.update(overrides)
        print(f"Params updated from sweep: {overrides}")
    if num_episodes is not None:
        hparams["num_episodes"] = num_episodes

    # trackio.init mirrors wandb.init. space_id (optional) hosts a free dashboard
    # on Hugging Face Spaces; without it the dashboard runs locally.
    run = wandb.init(
        project=PROJECT,
        name=run_name,
        config=hparams,
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
    )
    run_name = getattr(run, "name", None) or run_name or "run"

    # Create environment
    env = Env()

    # Fix random seed (for reproducibility)
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Number of inputs and actions
    n_inputs = 37
    n_actions = 12

    # Create policy and optimizer
    policy = Agent(n_inputs, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams["lr"])
    memory = ReplayMemory(hparams["replay_size"])

    # To resume from a checkpoint:
    # policy = torch.load('.../something_policy.pt', weights_only=False)
    # optimizer = torch.load('.../something_optimizer.pt', weights_only=False)

    action_std_decay = (
        -(hparams["std_min"] - hparams["std_init"])
        * hparams["log_interval"]
        / hparams["num_episodes"]
    )
    action_std_init = hparams["std_init"]
    action_std = action_std_init

    # Define the target_reward to stop the run before reaching num_episodes
    target_reward = 10000
    print(f"Target reward: {target_reward}")

    # Training loop
    running_reward = -100
    saving_reward = 0
    for i_episode in range(hparams["num_episodes"]):
        # Collect experience
        state, ep_reward, done = env.reset(), 0, False

        while not done:  # Don't infinite loop while learning
            action, a_logp, state_value = policy.compute_action(state, action_std)
            next_state, reward, done = env.step(action, render=False)

            if memory.store((state, action, a_logp, reward, next_state)):
                policy_loss, value_loss, avg_entropy, ratio = train(
                    policy, optimizer, memory, hparams, action_std
                )
                wandb.log(
                    {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "avg_reward": running_reward,
                        "avg_entropy": avg_entropy,
                        "ratio": ratio,
                    }
                )

            state = next_state
            ep_reward += reward

            if done:
                break

        # Update running reward
        running_reward = round(0.05 * ep_reward + (1 - 0.05) * running_reward, 2)

        # Log to check episode rewards and std
        if i_episode % hparams["log_interval"] == 0:
            print(
                f"Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}\tAction standard deviation: {action_std:.5f}"
            )
            action_std = action_std - action_std_decay
            action_std = round(action_std, 5)

        # Minimum reward needed to save a model and make a video
        if running_reward > 2000:
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
                print(f"Policy and Optimizer have been saved to {SAVE_DIR}")
                ep_reward = test(action_std, env, policy, i_episode)

        if running_reward > target_reward:
            print("Solved!")
            policy_path = os.path.join(
                SAVE_DIR, f"{run_name}_{i_episode}_Reward-{running_reward}_policy.pt"
            )
            optim_path = os.path.join(
                SAVE_DIR, f"{run_name}_{i_episode}_Reward-{running_reward}_optimizer.pt"
            )
            torch.save(policy, policy_path)
            torch.save(optimizer, optim_path)
            push_checkpoint_to_hub(policy_path)
            push_checkpoint_to_hub(optim_path)
            break

    print(f"Finished training! Running reward is now {running_reward}")
    wandb.finish()
    return running_reward


def run_sweep(count=50, episodes_per_trial=2000, seed=0):
    """Self-contained replacement for `wandb.sweep` + `wandb.agent`.
    Random-searches the `sweep_configuration` ranges; each trial is its own
    trackio run. trackio has no built-in sweep agent, so we drive it here."""
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
    parser = argparse.ArgumentParser(description="PPO train / sweep for ANYmal D")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run a random-search hyperparameter sweep instead of a single training run",
    )
    parser.add_argument(
        "--sweep-count", type=int, default=50, help="number of sweep trials"
    )
    parser.add_argument(
        "--sweep-episodes",
        type=int,
        default=2000,
        help="episodes per sweep trial (fewer than a full run)",
    )
    args = parser.parse_args()

    if args.sweep:
        run_sweep(count=args.sweep_count, episodes_per_trial=args.sweep_episodes)
    else:
        # Train with the specified hparams.
        train_or_sweep(is_sweep=False)
