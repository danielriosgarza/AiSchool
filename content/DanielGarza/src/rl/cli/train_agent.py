from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import os
import sys

import numpy as np

# Prefer local source (backend/src) over any installed package
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl.env import GeneralMicrobiomeEnv
from rl.wrappers import apply_observation_pipeline, EpsilonRandomActionWrapper
from rl.config import TopLevelConfig

from stable_baselines3 import SAC, PPO
try:
    from sb3_contrib import TQC  # type: ignore
    _HAS_TQC = True
except Exception:  # pragma: no cover - optional dependency
    TQC = None  # type: ignore
    _HAS_TQC = False
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def make_env(model_json: str, cfg: TopLevelConfig, seed: int):
    def _init():
        # Build base env and pre-reset to establish correct observation shape
        base_env = GeneralMicrobiomeEnv(model_json, cfg)
        base_env.reset(seed=seed)
        # Apply observation pipeline from config (normalize/clip/history)
        env = apply_observation_pipeline(base_env, cfg.observations.pipeline)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


class NormSyncCallback(BaseCallback):
    """Synchronize normalization stats from training env to eval env and save best stats."""

    def __init__(self, eval_env: DummyVecEnv, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        # get stats from training env
        try:
            train_stats_list = self.training_env.env_method("get_normalization_stats")
            train_stats = None
            for s in train_stats_list:
                if isinstance(s, dict) and s.get("status") not in {"normalization_not_enabled", "not_initialized"}:
                    train_stats = s
                    break
            if isinstance(train_stats, dict):
                # apply to eval env
                mean = np.array(train_stats["mean"], dtype=np.float32)
                m2 = np.array(train_stats["m2"], dtype=np.float32)
                count = int(train_stats["count"])
                self.eval_env.env_method("set_normalization_stats", mean, m2, count)
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        # save final normalization stats
        try:
            train_stats_list = self.training_env.env_method("get_normalization_stats")
            for s in train_stats_list:
                if isinstance(s, dict) and s.get("status") not in {"normalization_not_enabled", "not_initialized"}:
                    self.save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.save_path, "w") as f:
                        json.dump({
                            "mean": np.array(s["mean"], dtype=float).tolist(),
                            "m2": np.array(s["m2"], dtype=float).tolist(),
                            "count": int(s["count"]),
                        }, f)
                    break
        except Exception:
            pass


class NormFreezeCallback(BaseCallback):
    """Freeze observation normalization after a certain number of timesteps."""

    def __init__(self, freeze_after_steps: int):
        super().__init__(verbose=0)
        self.freeze_after_steps = int(freeze_after_steps)
        self._frozen = False

    def _on_step(self) -> bool:
        if not self._frozen and self.freeze_after_steps > 0 and self.num_timesteps >= self.freeze_after_steps:
            try:
                self.training_env.env_method("set_training", False)
                self._frozen = True
            except Exception:
                pass
        return True


class CheckpointCallback(BaseCallback):
    """Periodic model checkpointing (optionally saving normalization stats)."""

    def __init__(self, save_path: Path, save_freq: int, save_norm_stats: bool = False):
        super().__init__(verbose=0)
        self.save_path = Path(save_path)
        self.save_freq = int(save_freq)
        self.save_norm_stats = bool(save_norm_stats)
        self._last = 0

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.num_timesteps - self._last) >= self.save_freq:
            self._last = self.num_timesteps
            try:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt = self.save_path.with_name(self.save_path.name + f"_ckpt_{self.num_timesteps}.zip")
                self.model.save(str(ckpt))
                if self.save_norm_stats:
                    try:
                        stats_list = self.training_env.env_method("get_normalization_stats")
                        for s in stats_list:
                            if isinstance(s, dict) and s.get("status") not in {"normalization_not_enabled", "not_initialized"}:
                                stats_path = ckpt.with_name(ckpt.stem + "_normalization_stats.json")
                                with open(stats_path, "w") as f:
                                    json.dump({
                                        "mean": np.array(s["mean"], dtype=float).tolist(),
                                        "m2": np.array(s["m2"], dtype=float).tolist(),
                                        "count": int(s["count"]),
                                    }, f)
                                break
                    except Exception:
                        pass
            except Exception:
                pass
        return True


class TrainRewardBreakdownCallback(BaseCallback):
    """Aggregate per-episode reward breakdown during training and log to TensorBoard.

    Logs for each finished episode (per env):
      - train_bk/step_sum: sum of non-terminal terms over episode
      - train_bk/terminal_sum: sum of terminal terms at episode end
      - train_bk/ep_total: step_sum + terminal_sum (as seen via breakdown)
      - train_bk/ep_len: episode length (steps)
    """

    def __init__(self):
        super().__init__(verbose=0)
        self._step_sums = None  # type: ignore
        self._term_sums = None  # type: ignore
        self._ep_lens = None    # type: ignore

    def _on_training_start(self) -> None:
        try:
            n = int(self.training_env.num_envs)  # type: ignore[attr-defined]
        except Exception:
            n = 1
        self._step_sums = [0.0 for _ in range(n)]
        self._term_sums = [0.0 for _ in range(n)]
        self._ep_lens = [0 for _ in range(n)]

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos", [])  # type: ignore[assignment]
            dones = self.locals.get("dones", [])  # type: ignore[assignment]
            if not isinstance(infos, (list, tuple)):
                return True
            n = len(infos)
            # Ensure buffers are sized
            if self._step_sums is None or len(self._step_sums) != n:
                self._step_sums = [0.0 for _ in range(n)]
                self._term_sums = [0.0 for _ in range(n)]
                self._ep_lens = [0 for _ in range(n)]
            for i in range(n):
                info = infos[i] or {}
                rb = info.get("reward_breakdown", {}) or {}
                # accumulate step terms
                step_contrib = sum(float(v) for k, v in rb.items() if not str(k).startswith("terminal_"))
                term_contrib = sum(float(v) for k, v in rb.items() if str(k).startswith("terminal_"))
                self._step_sums[i] += float(step_contrib)
                self._term_sums[i] += float(term_contrib)
                self._ep_lens[i] += 1
                done = False
                try:
                    done = bool(dones[i])  # type: ignore[index]
                except Exception:
                    pass
                if done:
                    ep_total = float(self._step_sums[i] + self._term_sums[i])
                    self.logger.record("train_bk/step_sum", float(self._step_sums[i]))
                    self.logger.record("train_bk/terminal_sum", float(self._term_sums[i]))
                    self.logger.record("train_bk/ep_total", ep_total)
                    self.logger.record("train_bk/ep_len", int(self._ep_lens[i]))
                    # reset buffers for next episode
                    self._step_sums[i] = 0.0
                    self._term_sums[i] = 0.0
                    self._ep_lens[i] = 0
        except Exception:
            # Never break training due to diagnostics
            return True
        return True


def _load_config(config_path: Optional[str]) -> TopLevelConfig:
    if not config_path:
        return TopLevelConfig()
    path = Path(config_path)
    text = path.read_text()
    # Try YAML first if available, fallback to JSON
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    return TopLevelConfig(**data)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on GeneralMicrobiomeEnv")
    base = Path(__file__).resolve().parents[3]
    parser.add_argument("--algo", type=str, choices=["sac", "ppo", "tqc"], default="sac")
    parser.add_argument("--model-json", type=str, default=str(base / "modelTemplates" / "bh_bt_ri_complete_model_export.json"))
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config for TopLevelConfig")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--log-dir", type=str, default=str(base / "logs" / "rl"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=0, help="Checkpoint save frequency (timesteps); 0 disables")
    parser.add_argument("--save-norm-with-checkpoints", action="store_true", help="Also save normalization stats JSON alongside each checkpoint when --save-freq > 0")
    parser.add_argument("--freeze-norm-steps", type=int, default=0, help="Freeze observation normalization after N timesteps; 0 disables")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"], default="constant")
    parser.add_argument("--progress-bar", action="store_true", help="Show tqdm progress bar during training (SB3>=2.0)")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="Path for TensorBoard logs (enables SB3 TensorBoard logging)")
    parser.add_argument("--tb-log-name", type=str, default=None, help="TensorBoard run name (subfolder under --tensorboard-log)")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device for the policy (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--log-eval-breakdown", action="store_true", help="Log eval reward breakdown terms to TensorBoard at eval points")
    # Evaluation controls
    parser.add_argument("--eval-mode", type=str, choices=["fast", "balanced", "accurate"], default="fast", help="Simulation mode for evaluation env")
    parser.add_argument("--eval-mix", type=str, choices=["long_only", "same"], default="long_only", help="Evaluation episode mix: long_only (default) or same as training config")
    parser.add_argument("--eval-deterministic", dest="eval_deterministic", action="store_true", default=True, help="Use deterministic actions during evaluation (default: True)")
    parser.add_argument("--no-eval-deterministic", dest="eval_deterministic", action="store_false", help="Use stochastic actions during evaluation (deterministic=False)")

    # SAC-specific
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (SAC/PPO)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (SAC) or minibatch size (PPO)")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--ent-coef", type=str, default="auto_0.5")
    parser.add_argument("--target-entropy", type=float, default=None, help="SAC: override target entropy")
    parser.add_argument("--use-sde", action="store_true", default=True, help="SAC: enable State-Dependent Exploration")
    parser.add_argument("--no-use-sde", dest="use_sde", action="store_false", help="SAC: disable State-Dependent Exploration")
    parser.add_argument("--sde-sample-freq", type=int, default=16)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    # Optional epsilon-greedy exploration (training envs)
    parser.add_argument("--eps-start", type=float, default=0.0, help="Initial epsilon for epsilon-greedy exploration (training env)")
    parser.add_argument("--eps-end", type=float, default=0.0, help="Final epsilon for epsilon-greedy exploration")
    parser.add_argument("--eps-decay-steps", type=int, default=100000, help="Decay steps for epsilon-greedy exploration")

    # PPO-specific
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # Optional schedules and stability flags (parity with train_sac)
    parser.add_argument("--entropy-target-at", type=str, default=None, help="Comma-separated step:target list, e.g., 60000:-4,120000:-6")
    parser.add_argument("--sde-freeze-at", type=int, default=0, help="Freeze SDE noise (set sde_sample_freq=0) at this step (SAC only)")
    parser.add_argument("--lr-milestones", type=str, default=None, help="Comma-separated steps for LR decay, e.g., 60000,120000")
    parser.add_argument("--lr-factors", type=str, default=None, help="Comma-separated factors to multiply LR at milestones, e.g., 0.33,0.5")

    args = parser.parse_args()

    cfg = _load_config(args.config)
    model_json = args.model_json

    # Derive defaults that depend on algo
    if args.model_path is None:
        default_model_name = f"mg_rl_general_{args.algo}"
        args.model_path = str(Path(args.log_dir).parents[0] / "models" / default_model_name)
    if args.tb_log_name is None:
        args.tb_log_name = args.algo

    # Training environments
    env = DummyVecEnv([make_env(model_json, cfg, seed=args.seed + i) for i in range(args.n_envs)])
    # Optional epsilon-greedy exploration on training envs
    if float(args.eps_start) > 0.0:
        try:
            for i in range(args.n_envs):
                env.envs[i] = EpsilonRandomActionWrapper(
                    env.envs[i],
                    eps_start=float(args.eps_start),
                    eps_end=float(args.eps_end),
                    decay_steps=int(args.eps_decay_steps),
                    seed=args.seed + i,
                )
        except Exception:
            pass

    # Evaluation env using requested mode (default fast for parity)
    eval_cfg = TopLevelConfig(**cfg.model_dump())
    eval_cfg.episode.training_mode = args.eval_mode
    # Evaluation episode mixing behavior
    if str(args.eval_mix) == "long_only":
        try:
            mix = dict(eval_cfg.episode.mix)
            mix["enabled"] = True
            mix["short_prob"] = 0.0
            if "short_horizon" not in mix:
                mix["short_horizon"] = 1
            if "long_horizon" not in mix:
                src_mix = cfg.episode.mix or {}
                mix["long_horizon"] = int(src_mix.get("long_horizon", cfg.episode.horizon))
            eval_cfg.episode.mix = mix
        except Exception:
            pass
    eval_env = DummyVecEnv([make_env(model_json, eval_cfg, seed=123)])

    # learning rate schedule
    lr: Any
    if args.lr_schedule == "linear":
        try:
            from stable_baselines3.common.utils import get_linear_fn
            lr = get_linear_fn(args.learning_rate, args.learning_rate * 0.1)
        except Exception:
            lr = args.learning_rate
    else:
        lr = args.learning_rate

    # Build model per algorithm
    out = Path(args.model_path)
    norm_path = out.with_name(out.name + "_normalization_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.algo == "sac":
        model_kwargs: Dict[str, Any] = dict(
            learning_rate=lr,
            buffer_size=int(args.buffer_size),
            batch_size=int(args.batch_size),
            tau=5e-3,
            gamma=float(args.gamma),
            train_freq=int(args.train_freq),
            gradient_steps=int(args.gradient_steps),
            ent_coef=args.ent_coef,
            verbose=1,
            seed=int(args.seed),
            device=str(args.device),
            use_sde=bool(args.use_sde),
            tensorboard_log=args.tensorboard_log,
            sde_sample_freq=int(args.sde_sample_freq),
            learning_starts=int(args.learning_starts),
        )
        if args.target_entropy is not None:
            model_kwargs["target_entropy"] = float(args.target_entropy)
        model = SAC(
            "MlpPolicy",
            env,
            **model_kwargs,
        )
    elif args.algo == "tqc":
        if not _HAS_TQC:
            raise SystemExit("TQC requires sb3-contrib. Please install it: pip install sb3-contrib")
        model_kwargs = dict(
            learning_rate=lr,
            buffer_size=int(args.buffer_size),
            batch_size=int(args.batch_size),
            tau=5e-3,
            gamma=float(args.gamma),
            train_freq=int(args.train_freq),
            gradient_steps=int(args.gradient_steps),
            ent_coef=args.ent_coef,
            verbose=1,
            seed=int(args.seed),
            device=str(args.device),
            tensorboard_log=args.tensorboard_log,
            learning_starts=int(args.learning_starts),
        )
        if args.target_entropy is not None:
            model_kwargs["target_entropy"] = float(args.target_entropy)
        model = TQC(  # type: ignore[operator]
            "MlpPolicy",
            env,
            **model_kwargs,
        )
    else:
        # PPO
        model_kwargs = dict(
            learning_rate=lr,
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(0.0 if isinstance(args.ent_coef, str) else args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            verbose=1,
            seed=int(args.seed),
            device=str(args.device),
            tensorboard_log=args.tensorboard_log,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        if args.target_kl is not None:
            model_kwargs["target_kl"] = float(args.target_kl)
        model = PPO(
            "MlpPolicy",
            env,
            **model_kwargs,
        )

    # callbacks: eval + normalization sync
    callbacks = []
    
    # Custom EvalCallback that ensures proper normalization sync before evaluation
    class EvalCallbackWithNormSync(EvalCallback):
        def __init__(self, eval_env, norm_sync_callback, **kwargs):
            super().__init__(eval_env, **kwargs)
            self.norm_sync_callback = norm_sync_callback
            
        def _on_step(self) -> bool:
            # Force normalization sync before evaluation
            if (self.num_timesteps % self.eval_freq == 0) and (self.num_timesteps > self.n_eval_episodes):
                # Sync normalization stats before evaluation
                try:
                    train_stats_list = self.training_env.env_method("get_normalization_stats")
                    train_stats = None
                    for s in train_stats_list:
                        if isinstance(s, dict) and s.get("status") not in {"normalization_not_enabled", "not_initialized"}:
                            train_stats = s
                            break
                    if isinstance(train_stats, dict):
                        mean = np.array(train_stats["mean"], dtype=np.float32)
                        m2 = np.array(train_stats["m2"], dtype=np.float32)
                        count = int(train_stats["count"])
                        self.eval_env.env_method("set_normalization_stats", mean, m2, count)
                except Exception:
                    pass
            
            # Call parent's _on_step for evaluation
            return super()._on_step()
    
    # Add callback to save normalization stats with best model
    class BestModelNormStatsCallback(BaseCallback):
        def __init__(self, norm_stats_path: Path, best_model_dir: Path):
            super().__init__(verbose=0)
            self.norm_stats_path = Path(norm_stats_path)
            self.best_model_dir = Path(best_model_dir)
            self.best_model_dir.mkdir(parents=True, exist_ok=True)
            self._last_best_model_time = 0
            
        def _on_step(self) -> bool:
            # Check if best model was just saved by looking for the file
            best_model_path = self.best_model_dir / "best_model.zip"
            if best_model_path.exists():
                try:
                    current_time = best_model_path.stat().st_mtime
                    if current_time > self._last_best_model_time:
                        self._last_best_model_time = current_time
                        # Copy normalization stats to best model directory
                        if self.norm_stats_path.exists():
                            import shutil
                            best_norm_path = self.best_model_dir / "best_model_normalization_stats.json"
                            shutil.copy2(self.norm_stats_path, best_norm_path)
                            if self.verbose > 0:
                                print(f"Copied normalization stats to {best_norm_path}")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Failed to copy normalization stats: {e}")
            return True
    
    # Create norm sync callback first
    norm_sync_callback = NormSyncCallback(eval_env=eval_env, save_path=norm_path)
    
    # Create eval callback with norm sync
    eval_callback = EvalCallbackWithNormSync(
        eval_env=eval_env,
        norm_sync_callback=norm_sync_callback,
        best_model_save_path=str(out.parent),
        log_path=str(Path(args.log_dir)),
        eval_freq=max(1, int(args.eval_freq)),
        deterministic=bool(args.eval_deterministic),
        render=False
    )
    
    callbacks.append(norm_sync_callback)
    callbacks.append(BestModelNormStatsCallback(norm_path, out.parent))
    callbacks.append(eval_callback)
    callbacks.append(TrainRewardBreakdownCallback())
    if args.log_eval_breakdown:
        class _EvalBreakdownTB(BaseCallback):
            def __init__(self, eval_env: DummyVecEnv, freq: int):
                super().__init__(verbose=0)
                self.eval_env = eval_env
                self.freq = int(max(1, freq))
            def _on_step(self) -> bool:
                if (self.num_timesteps % self.freq) != 0:
                    return True
                try:
                    obs = self.eval_env.reset()
                    import numpy as _np
                    done = _np.array([False])
                    agg = {}
                    steps = 0
                    while not done[0]:
                        act, _ = self.model.predict(obs, deterministic=True)
                        obs, r, done, infos = self.eval_env.step(act)
                        info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
                        bd = info0.get("reward_breakdown", {})
                        for k, v in bd.items():
                            agg[k] = agg.get(k, 0.0) + float(v)
                        steps += 1
                    if steps > 0:
                        for k, v in agg.items():
                            self.logger.record(f"eval_bk/{k}", v / steps)
                            self.logger.record(f"eval_bk_sum/{k}", v)
                        self.logger.record("eval_bk/steps", steps)
                except Exception:
                    pass
                return True
        callbacks.append(_EvalBreakdownTB(eval_env=eval_env, freq=max(1, int(args.eval_freq))))
    if int(args.freeze_norm_steps) > 0:
        callbacks.append(NormFreezeCallback(freeze_after_steps=int(args.freeze_norm_steps)))
    if int(args.save_freq) > 0:
        callbacks.append(CheckpointCallback(save_path=out, save_freq=int(args.save_freq), save_norm_stats=bool(args.save_norm_with_checkpoints)))

    # Entropy target schedule (SAC/TQC)
    if args.algo in {"sac", "tqc"} and args.entropy_target_at:
        try:
            pairs = []
            for token in str(args.entropy_target_at).split(','):
                step_str, val_str = token.split(':')
                pairs.append((int(step_str), float(val_str)))
            pairs.sort()
            class _EntropyTargetCB(BaseCallback):
                def __init__(self, schedule):
                    super().__init__(0)
                    self.schedule = schedule
                    self.idx = 0
                def _on_step(self) -> bool:
                    while self.idx < len(self.schedule) and self.num_timesteps >= self.schedule[self.idx][0]:
                        _, target = self.schedule[self.idx]
                        try:
                            if hasattr(self.model, 'target_entropy'):
                                self.model.target_entropy = float(target)
                        except Exception:
                            pass
                        self.idx += 1
                    return True
            callbacks.append(_EntropyTargetCB(pairs))
        except Exception:
            pass

    # SDE freeze schedule (SAC only)
    if args.algo == "sac" and int(args.sde_freeze_at) > 0:
        class _SDEFreezeCB(BaseCallback):
            def __init__(self, step:int):
                super().__init__(0)
                self.step = int(step)
                self.done = False
            def _on_step(self) -> bool:
                if (not self.done) and self.num_timesteps >= self.step:
                    try:
                        if hasattr(self.model, 'sde_sample_freq'):
                            self.model.sde_sample_freq = 0
                    except Exception:
                        pass
                    self.done = True
                return True
        callbacks.append(_SDEFreezeCB(int(args.sde_freeze_at)))

    # Stepwise LR schedule (all algos)
    if args.lr_milestones and args.lr_factors:
        try:
            ms = [int(x) for x in str(args.lr_milestones).split(',') if x]
            fs = [float(x) for x in str(args.lr_factors).split(',') if x]
            if len(ms) == len(fs) and len(ms) > 0:
                class _LRCB(BaseCallback):
                    def __init__(self, milestones, factors):
                        super().__init__(0)
                        self.m = milestones
                        self.f = factors
                        self.idx = 0
                    def _set_lr(self, lr: float):
                        try:
                            if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                                for pg in self.model.actor.optimizer.param_groups:
                                    pg['lr'] = lr
                            if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                                for pg in self.model.critic.optimizer.param_groups:
                                    pg['lr'] = lr
                        except Exception:
                            pass
                    def _on_step(self) -> bool:
                        if self.idx < len(self.m) and self.num_timesteps >= self.m[self.idx]:
                            try:
                                current_lr = float(self.model.learning_rate)
                            except Exception:
                                current_lr = 3e-4
                            new_lr = max(1e-6, current_lr * float(self.f[self.idx]))
                            self._set_lr(new_lr)
                            try:
                                self.model.learning_rate = new_lr
                            except Exception:
                                pass
                            self.idx += 1
                        return True
                callbacks.append(_LRCB(ms, fs))
        except Exception:
            pass

    # Learn with optional progress bar (compatible across SB3 minor versions)
    if args.progress_bar:
        try:
            model.learn(total_timesteps=int(args.timesteps), callback=callbacks, progress_bar=True, tb_log_name=args.tb_log_name or args.algo)
        except TypeError:
            try:
                from stable_baselines3.common.callbacks import ProgressBarCallback  # type: ignore
                callbacks.append(ProgressBarCallback())
            except Exception:
                pass
            model.learn(total_timesteps=int(args.timesteps), callback=callbacks, tb_log_name=args.tb_log_name or args.algo)
    else:
        model.learn(total_timesteps=int(args.timesteps), callback=callbacks, tb_log_name=args.tb_log_name or args.algo)

    model.save(str(out))

    # save config alongside model
    with open(str(out) + "_config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()




