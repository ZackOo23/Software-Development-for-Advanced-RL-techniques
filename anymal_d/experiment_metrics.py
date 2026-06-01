"""Structured experiment metrics for ANYmal D training and evaluation.

This module is intentionally dependency-light so the repository can keep a
reproducible local record even when cloud trackers such as W&B, MLflow, or
Trackio are not available. Each run writes:

- runs/<run_name>/metrics.csv   -> spreadsheet-friendly metrics
- runs/<run_name>/metrics.jsonl -> one JSON object per episode
- runs/<run_name>/config.json   -> hyperparameters and metric definitions
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


METRIC_FIELDS: List[str] = [
    "timestamp_utc",
    "run_name",
    "phase",
    "episode",
    "global_step",
    "episode_return",
    "running_return",
    "episode_length",
    "episode_time_sec",
    "policy_loss",
    "value_loss",
    "entropy",
    "ratio",
    "action_std",
    "success",
    "success_rate",
    "fall",
    "fall_rate",
    "forward_distance_m",
    "path_length_m",
    "avg_forward_velocity_mps",
    "waypoint_completion_rate",
    "control_smoothness",
]


METRIC_DEFINITIONS: Dict[str, str] = {
    "episode_return": "Sum of rewards collected in the episode.",
    "running_return": "Exponential moving average used by the PPO trainer.",
    "episode_length": "Number of policy-control steps before termination.",
    "episode_time_sec": "Simulated time for the episode.",
    "policy_loss": "Most recent PPO actor loss.",
    "value_loss": "Most recent PPO critic loss.",
    "entropy": "Most recent policy entropy.",
    "ratio": "Most recent PPO probability ratio mean.",
    "success": "1 when the robot does not fall and reaches success_distance_m.",
    "success_rate": "Cumulative success fraction for the current phase.",
    "fall": "1 when the episode terminates because base height falls below threshold.",
    "fall_rate": "Cumulative fall fraction for the current phase.",
    "forward_distance_m": "Net forward displacement in meters, final x - start x.",
    "path_length_m": "Approximate absolute x-distance travelled during the episode.",
    "avg_forward_velocity_mps": "forward_distance_m / episode_time_sec.",
    "waypoint_completion_rate": "Fraction of configured forward waypoints reached.",
    "control_smoothness": "Mean squared change between consecutive bounded actions. Lower is smoother.",
}


def _safe_run_name(run_name: Optional[str]) -> str:
    name = run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "run"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        import numpy as np  # optional; available in this repo

        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except Exception:
        pass
    return value


class MetricsLogger:
    """Write episode-level metrics to CSV and JSONL with cumulative rates."""

    def __init__(
        self,
        project_root: str,
        run_name: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name = _safe_run_name(run_name)
        self.run_dir = os.path.join(project_root, "runs", self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

        self._phase_counts: Dict[str, Dict[str, float]] = {}
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()

        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": self.run_name,
            "metrics_csv": self.csv_path,
            "metrics_jsonl": self.jsonl_path,
            "config": config or {},
            "metric_definitions": METRIC_DEFINITIONS,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_safe)

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()

    def _update_rates(self, phase: str, success: bool, fall: bool) -> Dict[str, float]:
        counts = self._phase_counts.setdefault(
            phase, {"episodes": 0.0, "successes": 0.0, "falls": 0.0}
        )
        counts["episodes"] += 1.0
        counts["successes"] += float(success)
        counts["falls"] += float(fall)
        return {
            "success_rate": counts["successes"] / counts["episodes"],
            "fall_rate": counts["falls"] / counts["episodes"],
        }

    def build_episode_record(
        self,
        *,
        phase: str,
        episode: int,
        global_step: int,
        episode_return: float,
        running_return: Optional[float],
        episode_length: int,
        episode_time_sec: float,
        fall: bool,
        forward_distance_m: float,
        path_length_m: float,
        control_smoothness: float,
        success_distance_m: float,
        waypoint_distances_m: Iterable[float],
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        ratio: Optional[float] = None,
        action_std: Optional[float] = None,
    ) -> Dict[str, Any]:
        success = (not fall) and (forward_distance_m >= success_distance_m)
        waypoints = list(waypoint_distances_m)
        if waypoints:
            waypoint_completion_rate = sum(
                1 for w in waypoints if forward_distance_m >= float(w)
            ) / len(waypoints)
        else:
            waypoint_completion_rate = None

        rates = self._update_rates(phase, success, fall)
        avg_forward_velocity = (
            forward_distance_m / episode_time_sec if episode_time_sec > 0 else 0.0
        )

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": self.run_name,
            "phase": phase,
            "episode": int(episode),
            "global_step": int(global_step),
            "episode_return": float(episode_return),
            "running_return": None if running_return is None else float(running_return),
            "episode_length": int(episode_length),
            "episode_time_sec": float(episode_time_sec),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "ratio": ratio,
            "action_std": action_std,
            "success": int(success),
            "success_rate": rates["success_rate"],
            "fall": int(fall),
            "fall_rate": rates["fall_rate"],
            "forward_distance_m": float(forward_distance_m),
            "path_length_m": float(path_length_m),
            "avg_forward_velocity_mps": float(avg_forward_velocity),
            "waypoint_completion_rate": waypoint_completion_rate,
            "control_smoothness": float(control_smoothness),
        }

    def log(self, record: Dict[str, Any]) -> None:
        clean = {field: _json_safe(record.get(field)) for field in METRIC_FIELDS}
        self._writer.writerow({k: "" if v is None else v for k, v in clean.items()})
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(clean, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    @staticmethod
    def tracker_payload(record: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Return only numeric values for Trackio/W&B-style logging."""
        payload: Dict[str, float] = {}
        for key, value in record.items():
            value = _json_safe(value)
            if isinstance(value, bool):
                payload[f"{prefix}{key}"] = float(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[f"{prefix}{key}"] = float(value)
        return payload
