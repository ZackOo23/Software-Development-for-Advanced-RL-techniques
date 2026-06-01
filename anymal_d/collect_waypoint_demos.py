"""
Collect waypoint demonstrations for the ANYmal D MuJoCo PPO repository.

Run from the repository root:
    python anymal_d/collect_waypoint_demos.py --episodes 5 --policy auto
    python anymal_d/collect_waypoint_demos.py --episodes 1 --policy auto --live
    python anymal_d/collect_waypoint_demos.py --episodes 3 --source scripted --live

Output:
    datasets/waypoint_demos/<run_name>/demos.npz
    datasets/waypoint_demos/<run_name>/summary.json
    datasets/waypoint_demos/<run_name>/episodes.csv
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Import the existing environment and checkpoint utilities from the PPO file.
from RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING import (
    Agent,
    Env,
    N_ACT,
    N_OBS,
    MUJOCO_STEPS,
    PROJECT_ROOT,
    SAVE_DIR,
    pick_latest_checkpoint,
)


DEFAULT_WAYPOINTS = [
    [0.35, 0.00],
    [0.70, 0.00],
    [1.05, 0.00],
    [1.40, 0.00],
    [1.75, 0.00],
]


def wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Convert MuJoCo base quaternion [qw, qx, qy, qz] to yaw."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def robot_xy_yaw(env: Env):
    """Return robot world x, y, yaw from the MuJoCo qpos."""
    x = float(env.data.qpos[0])
    y = float(env.data.qpos[1])
    qw, qx, qy, qz = [float(v) for v in env.data.qpos[3:7]]
    yaw = quat_to_yaw(qw, qx, qy, qz)
    return x, y, yaw


def make_goal_observation(obs: np.ndarray, env: Env, waypoints: np.ndarray, wp_index: int) -> np.ndarray:
    """
    Add waypoint context to the normal PPO observation.

    obs_goal = [normal_obs(35), dx_body, dy_body, distance, heading_error, wp_progress]
    This is useful later for imitation learning because the policy can know
    which waypoint it should move toward.
    """
    x, y, yaw = robot_xy_yaw(env)
    target = waypoints[min(wp_index, len(waypoints) - 1)]
    dx_world = float(target[0] - x)
    dy_world = float(target[1] - y)

    # Rotate world error into the robot/body frame.
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    dx_body = c * dx_world - s * dy_world
    dy_body = s * dx_world + c * dy_world

    distance = math.sqrt(dx_world * dx_world + dy_world * dy_world)
    desired_yaw = math.atan2(dy_world, dx_world)
    heading_error = wrap_angle(desired_yaw - yaw)
    wp_progress = float(wp_index) / max(len(waypoints) - 1, 1)

    goal = np.array([dx_body, dy_body, distance, heading_error, wp_progress], dtype=np.float32)
    return np.concatenate([obs.astype(np.float32), goal], axis=0)


def waypoint_status(env: Env, waypoints: np.ndarray, wp_index: int, reach_radius: float):
    """Check if the current waypoint was reached and advance the index."""
    x, y, _ = robot_xy_yaw(env)
    target = waypoints[min(wp_index, len(waypoints) - 1)]
    distance = float(np.linalg.norm(target - np.array([x, y], dtype=np.float32)))
    reached_now = distance <= reach_radius
    if reached_now and wp_index < len(waypoints) - 1:
        wp_index += 1
    return wp_index, reached_now, distance


def load_policy(policy_arg: str):
    """Load a PPO checkpoint saved by the existing repo."""
    if policy_arg == "auto":
        policy_path = pick_latest_checkpoint(SAVE_DIR)
    else:
        policy_path = policy_arg
    print(f"Loading demonstration policy from: {policy_path}")
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    policy.eval()
    return policy, policy_path


def scripted_waypoint_action(env: Env, wp_distance: float, wp_heading_error: float, step_id: int) -> np.ndarray:
    """
    Simple open-loop trot-like controller.

    This is mainly a fallback to test dataset generation if no PPO checkpoint exists.
    For better imitation data, use --source policy with a trained checkpoint.
    """
    phase = step_id * 0.12
    speed_cmd = float(np.clip(wp_distance, 0.0, 1.0))
    turn_cmd = float(np.clip(wp_heading_error, -0.7, 0.7))

    # Bounded action before Env.step applies tanh. Joint order:
    # LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
    # LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
    bounded = np.zeros(N_ACT, dtype=np.float32)
    leg_offsets = [0, math.pi, math.pi, 0]  # diagonal pairs: LF/RH, RF/LH
    haa_turn_signs = [-1.0, -1.0, 1.0, 1.0]

    for leg in range(4):
        base = 3 * leg
        ph = phase + leg_offsets[leg]
        swing = math.sin(ph)
        lift = max(0.0, math.sin(ph))

        bounded[base + 0] = 0.12 * haa_turn_signs[leg] * turn_cmd
        bounded[base + 1] = 0.45 * speed_cmd * swing
        bounded[base + 2] = 0.35 * speed_cmd * lift - 0.10 * speed_cmd

    # Convert desired bounded action into raw action because Env.step does tanh(raw_action).
    bounded = np.clip(bounded, -0.85, 0.85)
    raw_action = np.arctanh(bounded).astype(np.float32)
    return raw_action


def parse_waypoints(text: str) -> np.ndarray:
    """
    Parse waypoints from a string like: "0.4,0;0.8,0;1.2,0.2".
    """
    if not text:
        return np.array(DEFAULT_WAYPOINTS, dtype=np.float32)
    points = []
    for item in text.split(";"):
        x_str, y_str = item.split(",")
        points.append([float(x_str.strip()), float(y_str.strip())])
    if len(points) < 1:
        raise ValueError("At least one waypoint is required.")
    return np.array(points, dtype=np.float32)


def collect(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    waypoints = parse_waypoints(args.waypoints)
    run_name = args.run_name or datetime.now().strftime("waypoint_demos_%Y%m%d_%H%M%S")
    out_dir = Path(PROJECT_ROOT) / "datasets" / "waypoint_demos" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = Env(fall_threshold=args.fall_threshold)
    if args.live:
        env.attach_viewer()

    policy = None
    policy_path = None
    if args.source == "policy":
        policy, policy_path = load_policy(args.policy)

    obs_records = []
    obs_goal_records = []
    action_records = []
    next_obs_records = []
    next_obs_goal_records = []
    reward_records = []
    done_records = []
    episode_records = []
    step_records = []
    robot_xy_records = []
    waypoint_xy_records = []
    waypoint_index_records = []
    reached_records = []
    fall_records = []

    episode_summaries = []
    global_step = 0

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        wp_index = 0
        reached_count = 0
        last_distance = 0.0
        last_info = {"fall": False}

        for step_id in range(args.max_steps):
            obs_goal = make_goal_observation(obs, env, waypoints, wp_index)
            x, y, yaw = robot_xy_yaw(env)
            target = waypoints[min(wp_index, len(waypoints) - 1)]
            dx = float(target[0] - x)
            dy = float(target[1] - y)
            wp_distance = math.sqrt(dx * dx + dy * dy)
            wp_heading_error = wrap_angle(math.atan2(dy, dx) - yaw)

            if args.source == "policy":
                action, _, _ = policy.compute_action(obs)
            else:
                action = scripted_waypoint_action(env, wp_distance, wp_heading_error, step_id)

            next_obs, reward, done, last_info = env.step(action, render=args.live)
            next_wp_index, reached_now, last_distance = waypoint_status(
                env, waypoints, wp_index, args.reach_radius
            )
            if reached_now:
                reached_count += 1
            wp_index = next_wp_index

            # Extra waypoint reward saved only as metadata for IL analysis.
            # It does not change the MuJoCo/PPO environment reward.
            waypoint_bonus = 1.0 if reached_now else 0.0
            shaped_reward = float(reward + waypoint_bonus)
            next_obs_goal = make_goal_observation(next_obs, env, waypoints, wp_index)

            obs_records.append(obs.astype(np.float32))
            obs_goal_records.append(obs_goal.astype(np.float32))
            action_records.append(np.asarray(action, dtype=np.float32))
            next_obs_records.append(next_obs.astype(np.float32))
            next_obs_goal_records.append(next_obs_goal.astype(np.float32))
            reward_records.append(shaped_reward)
            done_records.append(bool(done))
            episode_records.append(ep)
            step_records.append(step_id)
            robot_xy_records.append([x, y])
            waypoint_xy_records.append(target.astype(np.float32))
            waypoint_index_records.append(wp_index)
            reached_records.append(bool(reached_now))
            fall_records.append(bool(last_info.get("fall", False)))

            obs = next_obs
            ep_return += float(reward)
            global_step += 1

            if done or reached_count >= len(waypoints):
                break

        completion_rate = reached_count / len(waypoints)
        ep_summary = {
            "episode": ep,
            "steps": step_id + 1,
            "return": ep_return,
            "reached_waypoints": reached_count,
            "total_waypoints": len(waypoints),
            "waypoint_completion_rate": completion_rate,
            "final_waypoint_distance_m": float(last_distance),
            "fall": bool(last_info.get("fall", False)),
        }
        episode_summaries.append(ep_summary)
        print(
            f"Episode {ep:03d} | steps={ep_summary['steps']} | "
            f"return={ep_return:.2f} | waypoints={reached_count}/{len(waypoints)} | "
            f"fall={ep_summary['fall']}"
        )

    if args.live:
        env.detach_viewer()

    npz_path = out_dir / "demos.npz"
    np.savez_compressed(
        npz_path,
        obs=np.asarray(obs_records, dtype=np.float32),
        obs_goal=np.asarray(obs_goal_records, dtype=np.float32),
        actions=np.asarray(action_records, dtype=np.float32),
        next_obs=np.asarray(next_obs_records, dtype=np.float32),
        next_obs_goal=np.asarray(next_obs_goal_records, dtype=np.float32),
        rewards=np.asarray(reward_records, dtype=np.float32),
        dones=np.asarray(done_records, dtype=np.bool_),
        episode=np.asarray(episode_records, dtype=np.int32),
        step=np.asarray(step_records, dtype=np.int32),
        robot_xy=np.asarray(robot_xy_records, dtype=np.float32),
        waypoint_xy=np.asarray(waypoint_xy_records, dtype=np.float32),
        waypoint_index=np.asarray(waypoint_index_records, dtype=np.int32),
        reached=np.asarray(reached_records, dtype=np.bool_),
        fall=np.asarray(fall_records, dtype=np.bool_),
        waypoints=waypoints.astype(np.float32),
    )

    summary = {
        "run_name": run_name,
        "source": args.source,
        "policy_path": policy_path,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "reach_radius": args.reach_radius,
        "fall_threshold": args.fall_threshold,
        "mujo_co_steps_per_action": MUJOCO_STEPS,
        "obs_dim": N_OBS,
        "obs_goal_dim": N_OBS + 5,
        "action_dim": N_ACT,
        "waypoints": waypoints.tolist(),
        "total_samples": len(action_records),
        "mean_completion_rate": float(np.mean([e["waypoint_completion_rate"] for e in episode_summaries])) if episode_summaries else 0.0,
        "fall_rate": float(np.mean([e["fall"] for e in episode_summaries])) if episode_summaries else 0.0,
        "episodes_detail": episode_summaries,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_summaries[0].keys())) if episode_summaries else None
        if writer:
            writer.writeheader()
            writer.writerows(episode_summaries)

    print("\nDataset saved:")
    print(f"  {npz_path}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'episodes.csv'}")
    print(f"Samples: {len(action_records)}")
    print(f"Mean waypoint completion: {summary['mean_completion_rate']:.2f}")
    print(f"Fall rate: {summary['fall_rate']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ANYmal D waypoint demonstration dataset.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["policy", "scripted"], default="policy")
    parser.add_argument("--policy", type=str, default="auto", help="Path to *_policy.pt checkpoint or 'auto'.")
    parser.add_argument("--waypoints", type=str, default="", help="Example: '0.4,0;0.8,0;1.2,0.2'")
    parser.add_argument("--reach-radius", type=float, default=0.18)
    parser.add_argument("--fall-threshold", type=float, default=0.35)
    parser.add_argument("--live", action="store_true", help="Open MuJoCo viewer while collecting demos.")
    args = parser.parse_args()
    collect(args)
