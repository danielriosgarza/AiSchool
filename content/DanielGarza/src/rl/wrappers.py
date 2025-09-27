from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, clip_range: Tuple[float, float] = (-5.0, 5.0), epsilon: float = 1e-8, update_freq: int = 1):
        super().__init__(env)
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.obs_mean = None
        self._m2 = None
        self.obs_std = None
        self.obs_count = 0
        self.training = True
        low, high = clip_range
        self.observation_space = spaces.Box(low=low, high=high, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_mean is None:
            self.obs_mean = np.zeros_like(obs, dtype=np.float32)
            self._m2 = np.zeros_like(obs, dtype=np.float32)
            self.obs_std = np.ones_like(obs, dtype=np.float32)
        if self.training and (self.obs_count % self.update_freq == 0):
            self._update(obs)
        std = self._std()
        x = (obs - self.obs_mean) / (std + self.epsilon)
        x = np.clip(x, self.clip_range[0], self.clip_range[1])
        self.obs_std = std.astype(np.float32)
        return x.astype(np.float32)

    def _update(self, x):
        self.obs_count += 1
        delta = x - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = x - self.obs_mean
        self._m2 += delta * delta2

    def _std(self):
        if self.obs_count > 1:
            var = self._m2 / (self.obs_count - 1)
            return np.sqrt(np.maximum(var, self.epsilon))
        return np.ones_like(self.obs_mean, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def set_training(self, training: bool):
        self.training = bool(training)

    def get_normalization_stats(self) -> Dict[str, Any]:
        if self.obs_mean is None:
            return {"status": "not_initialized"}
        return {"count": self.obs_count, "mean": self.obs_mean.copy(), "m2": self._m2.copy(), "std": self.obs_std.copy(), "clip_range": self.clip_range, "training": self.training}

    def set_normalization_stats(self, mean: np.ndarray, m2: np.ndarray, count: int):
        self.obs_mean = mean.astype(np.float32).copy()
        self._m2 = m2.astype(np.float32).copy()
        self.obs_count = int(count)
        self.obs_std = self._std().astype(np.float32)



class HistoryStackWrapper(gym.ObservationWrapper):
    """Stacks the last N observation frames along the feature axis.

    Observation shape becomes (N * original_dim,).
    """

    def __init__(self, env: gym.Env, frames: int = 4):
        super().__init__(env)
        assert frames >= 1
        self.frames = int(frames)
        self._buf: deque[np.ndarray] = deque(maxlen=self.frames)
        low = np.repeat(env.observation_space.low, self.frames, axis=0)
        high = np.repeat(env.observation_space.high, self.frames, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf.clear()
        for _ in range(self.frames):
            self._buf.append(np.zeros_like(obs, dtype=np.float32))
        self._buf.append(obs.astype(np.float32))
        return self._get_stacked(), info

    def observation(self, obs):
        self._buf.append(obs.astype(np.float32))
        return self._get_stacked()

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self._buf), axis=0).astype(np.float32)


class ClipWrapper(gym.ObservationWrapper):
    """Clips observations to [min, max] independently of normalization."""

    def __init__(self, env: gym.Env, clip_min: float = -10.0, clip_max: float = 10.0):
        super().__init__(env)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        low = np.full(env.observation_space.shape, self.clip_min, dtype=np.float32)
        high = np.full(env.observation_space.shape, self.clip_max, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.clip(obs, self.clip_min, self.clip_max).astype(np.float32)


def apply_observation_pipeline(env: gym.Env, pipeline: List[Dict[str, Any]] | None) -> gym.Env:
    """Apply observation pipeline steps (normalize, clip, history_stack) as wrappers.

    Recognized steps:
      - {normalize: {method: "running_mean_var", clip?: [min, max], update_freq?, epsilon?}}
      - {clip: {min: <float>, max: <float>}}
      - {history_stack: {frames: <int>}}
    """
    if not pipeline:
        return env

    norm_clip: Tuple[float, float] | None = None
    norm_cfg: Dict[str, Any] = {}
    for step in pipeline:
        if not isinstance(step, dict) or len(step) != 1:
            continue
        (key, val), = step.items()
        if key == "normalize":
            method = (val or {}).get("method", "running_mean_var")
            if method != "running_mean_var":
                continue
            norm_cfg = val or {}
            cr = norm_cfg.get("clip")
            if isinstance(cr, (list, tuple)) and len(cr) == 2:
                norm_clip = (float(cr[0]), float(cr[1]))

    if norm_cfg:
        clip_range = norm_clip if norm_clip is not None else (-5.0, 5.0)
        env = ObservationNormalizationWrapper(
            env,
            clip_range=(float(clip_range[0]), float(clip_range[1])),
            epsilon=float(norm_cfg.get("epsilon", 1e-8)),
            update_freq=int(norm_cfg.get("update_freq", 1)),
        )

    for step in pipeline:
        if not isinstance(step, dict) or len(step) != 1:
            continue
        (key, val), = step.items()
        val = val or {}
        if key == "clip":
            lo = float(val.get("min", -10.0))
            hi = float(val.get("max", 10.0))
            env = ClipWrapper(env, clip_min=lo, clip_max=hi)
        elif key == "history_stack":
            frames = int(val.get("frames", 1))
            if frames > 1:
                env = HistoryStackWrapper(env, frames=frames)

    return env


class EpsilonRandomActionWrapper(gym.ActionWrapper):
    """Epsilon-greedy exploration wrapper for continuous actions.

    With probability eps(t) returns a random action from the env's action_space,
    otherwise forwards the agent's action. eps decays linearly from eps_start to
    eps_end over decay_steps environment steps.
    """

    def __init__(self, env: gym.Env, eps_start: float = 0.1, eps_end: float = 0.0, decay_steps: int = 100_000, seed: int | None = None):
        super().__init__(env)
        self.eps_start = float(max(0.0, eps_start))
        self.eps_end = float(max(0.0, eps_end))
        self.decay_steps = int(max(1, decay_steps))
        self._steps = 0
        self._rng = np.random.default_rng(seed)
        self.training = True

    def action(self, action):
        if (not self.training) or self.eps_start <= 0.0:
            return action
        # Linear decay
        frac = min(1.0, self._steps / float(self.decay_steps))
        eps = (1.0 - frac) * self.eps_start + frac * self.eps_end
        self._steps += 1
        if self._rng.random() < eps:
            return self.env.action_space.sample()
        return action

    # Allow toggling from callbacks/CLI
    def set_training(self, training: bool):
        self.training = bool(training)
