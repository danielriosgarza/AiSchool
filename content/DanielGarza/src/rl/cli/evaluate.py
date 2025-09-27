from __future__ import annotations

from pathlib import Path
import json
import argparse
import sys

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Prefer local source (backend/src) over any installed package
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl.env import GeneralMicrobiomeEnv
from rl.wrappers import ObservationNormalizationWrapper, apply_observation_pipeline
from rl.config import TopLevelConfig


def make_env(model_json: str, cfg: TopLevelConfig, seed: int):
    def _init():
        base_env = GeneralMicrobiomeEnv(model_json, cfg)
        base_env.reset(seed=seed)
        env = apply_observation_pipeline(base_env, cfg.observations.pipeline)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def _load_config(config_path: str | None) -> TopLevelConfig:
    if not config_path:
        cfg = TopLevelConfig()
    else:
        text = Path(config_path).read_text()
        try:
            import yaml  # type: ignore
            cfg = TopLevelConfig(**yaml.safe_load(text))
        except Exception:
            cfg = TopLevelConfig(**json.loads(text))
    cfg.episode.training_mode = "accurate"
    return cfg


def main():
    base = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model")
    parser.add_argument("--algo", type=str, choices=["sac", "ppo"], default="sac")
    parser.add_argument("--model-json", type=str, default=str(base / "examples" / "bh_bt_ri_complete_model_export.json"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--norm-stats", type=str, default=None, help="Path to normalization stats JSON (auto if omitted)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    env = DummyVecEnv([make_env(args.model_json, cfg, seed=args.seed)])

    # default model path if not provided
    if args.model_path is None:
        default_name = f"mg_rl_general_{args.algo}.zip"
        args.model_path = str(base / "models" / default_name)
    mp = Path(args.model_path)
    if not mp.exists():
        print("No model found; run training first.")
        return
    if args.algo == "ppo":
        model = PPO.load(str(mp))
    else:
        model = SAC.load(str(mp))
    # try to load normalization stats
    norm_path = Path(args.norm_stats) if args.norm_stats else mp.with_name(mp.stem + "_normalization_stats.json")
    try:
        if norm_path.exists():
            saved = json.loads(norm_path.read_text())
            mean = np.array(saved["mean"], dtype=np.float32)
            m2 = np.array(saved["m2"], dtype=np.float32)
            count = int(saved["count"])
            env.env_method("set_normalization_stats", mean, m2, count)
    except Exception:
        pass
    obs = env.reset()
    done = np.array([False])
    total = 0.0
    while not done[0]:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = env.step(act)
        total += float(r[0])
    print(f"Evaluation reward: {total:.3f}")


if __name__ == "__main__":
    main()


