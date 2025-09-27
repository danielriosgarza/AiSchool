#!/usr/bin/env python3
"""
Publication-quality plotting for the general RL framework (mg_rl_general).

Runs a trained SAC policy on GeneralMicrobiomeEnv for a specified simulated
duration and generates high-quality figures:
 - Target metabolite concentration over time
 - pH used trajectory
 - Per-step reward and cumulative reward
 - Control actions time series (q, v, [pH_ctrl, pH_set], stir, temp)
 - Kinetics replay (metabolites, pH, active subpopulations, species aggregates)

Usage (example):
  python src/rl/cli/plot_general_controller.py \
      --model-json examples/bh_bt_ri_complete_model_export.json \
      --config src/rl/examples/configs/acetate_control.yaml \
      --model-path models/rl_acetate_sac.zip \
      --hours 500 \
      --det \
      --out plots/rl_acetate

Notes:
 - Uses the general framework’s config (actions/observations/rewards).
 - Auto-loads normalization stats if alongside the model unless disabled.
 - Saves PNG (300 dpi) and PDF versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
try:
    from sb3_contrib import TQC  # type: ignore
    _HAS_TQC = True
except Exception:  # pragma: no cover - optional dependency
    TQC = None  # type: ignore
    _HAS_TQC = False

import datetime as _dt

# Prefer local source (backend/src) over installed package
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl.env import GeneralMicrobiomeEnv
from rl.config import TopLevelConfig
from rl.wrappers import apply_observation_pipeline, ObservationNormalizationWrapper
from rl.model_adapter import ModelAdapter, AdapterSettings
import csv


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    w = int(w)
    pad = w - 1
    ext = np.r_[np.full(pad, x[0]), x]
    c = np.convolve(ext, np.ones(w) / w, mode="valid")
    return c[: x.size]


def _load_config(config_path: Optional[str]) -> TopLevelConfig:
    if not config_path:
        cfg = TopLevelConfig()
    else:
        text = Path(config_path).read_text()
        try:
            import yaml  # type: ignore
            cfg = TopLevelConfig(**yaml.safe_load(text))
        except Exception:
            cfg = TopLevelConfig(**json.loads(text))
    return cfg


def _try_load_norm_stats(env: Any, model_path: Path, norm_stats_path: Optional[Path], enable: bool = True) -> None:
    if not enable:
        return
    path = None
    if norm_stats_path is not None:
        path = norm_stats_path
    else:
        cand = model_path.with_name(model_path.stem + "_normalization_stats.json")
        if cand.exists():
            path = cand
    if path is None or not path.exists():
        return
    try:
        saved = json.loads(path.read_text())
        mean = np.array(saved["mean"], dtype=np.float32)
        m2 = np.array(saved["m2"], dtype=np.float32)
        count = int(saved["count"])
        # Walk wrappers to find ObservationNormalizationWrapper
        cur = env
        target = None
        while cur is not None:
            if isinstance(cur, ObservationNormalizationWrapper):
                target = cur
                break
            cur = getattr(cur, "env", None)
        if target is not None:
            target.set_normalization_stats(mean, m2, count)
    except Exception:
        pass


def _collect_run(
    model: Optional[Any],
    env_wrapped: Any,
    env_base: GeneralMicrobiomeEnv,
    cfg: TopLevelConfig,
    hours: float,
    det: bool,
    seed: int,
    random_actions: bool = False,
) -> Dict[str, Any]:
    obs, info = env_wrapped.reset(seed=seed)
    include_pH = env_base._pH_mode in ("switchable", "controlled")
    dt = float(cfg.episode.dt_hours)
    n_max = int(max(1, np.ceil(hours / max(1e-9, dt))))

    t = []
    reward = []
    cum_reward = []
    pH_used = []
    target = []
    actions_decoded: List[Dict[str, float]] = []

    # Determine target name from config if explicitly set; otherwise, do not assume a specific metabolite
    tgt = getattr(cfg, "target", None) or {}
    target_name = None
    target_type = "metabolite"
    if isinstance(tgt, dict):
        raw_name = tgt.get("name")
        if isinstance(raw_name, str) and raw_name:
            target_name = raw_name
        raw_type = tgt.get("type")
        if isinstance(raw_type, str) and raw_type:
            target_type = raw_type.lower()
    elif isinstance(tgt, str) and tgt:
        target_name = str(tgt)

    target_label = "target"
    if isinstance(target_name, str) and target_name:
        target_label = f"{target_name} biomass" if target_type == "biomass" else target_name

    target_idx = None
    species_idx = None
    species_order: List[str] = []
    if isinstance(target_name, str) and target_name:
        if target_type == "biomass":
            species_order = list(getattr(env_base._obs_builder, "_species_order", []))
            try:
                species_idx = species_order.index(target_name)
            except ValueError:
                species_idx = None
        else:
            try:
                target_idx = int(env_base.adapter.get_metabolite_index(target_name))
                target_type = "metabolite"
            except Exception:
                target_idx = None

    total = 0.0
    for step in range(n_max):
        if random_actions:
            act = env_wrapped.action_space.sample()
        else:
            act, _ = model.predict(obs, deterministic=det)
        # decode for logging
        decoded = env_base.schema.decode(act, include_pH=include_pH)
        actions_decoded.append({k: float(decoded.get(k, 0.0)) for k in ("q", "v", "pH_ctrl", "pH_set", "stir", "temp")})
        obs, r, terminated, truncated, info = env_wrapped.step(act)
        total += float(r)

        t.append(float(env_base.t))
        reward.append(float(r))
        cum_reward.append(float(total))

        mets_now = env_base._prev_metabolites.copy() if env_base._prev_metabolites is not None else env_base.adapter.metabolome.get_concentration().copy()
        # pH used
        if include_pH and int(decoded.get("pH_ctrl", 0)) == 1:
            pH_used.append(float(decoded.get("pH_set", 0.0)))
        else:
            pH_used.append(float(env_base.adapter.compute_emergent_pH(mets_now)))
        # target
        target_val = float("nan")
        if target_type == "metabolite" and isinstance(target_idx, int):
            try:
                target_val = float(mets_now[target_idx])
            except Exception:
                target_val = float("nan")
        elif target_type == "biomass" and isinstance(species_idx, int):
            try:
                state = env_base.adapter.reactor.get_states()
                live = env_base._obs_builder.species_live_signals(state)
                n_species = len(species_order)
                if n_species > 0 and live.size >= n_species:
                    target_val = float(live[species_idx])
            except Exception:
                target_val = float("nan")
        target.append(target_val)

        if float(env_base.t) >= float(hours):
            break
        if bool(terminated) or bool(truncated):
            break

    # Also capture final dt used
    return {
        "t": np.array(t, dtype=np.float32),
        "reward": np.array(reward, dtype=np.float32),
        "cum_reward": np.array(cum_reward, dtype=np.float32),
        "pH": np.array(pH_used, dtype=np.float32),
        "target": np.array(target, dtype=np.float32),
        "dt": dt,
        "actions": actions_decoded,
        "include_pH": include_pH,
        "target_name": target_label,
        "target_kind": target_type,
    }


def _replay_kinetics(
    model_json: str,
    cfg: TopLevelConfig,
    actions: List[Dict[str, float]],
    pH_mode: str,
) -> Dict[str, Any]:
    adapter = ModelAdapter(
        model_json,
        AdapterSettings(
            dt_hours=cfg.episode.dt_hours,
            min_steps_per_pulse=cfg.simulation.min_steps_per_pulse,
            steps_per_hour_factor=cfg.simulation.steps_per_hour_factor,
            training_mode=cfg.episode.training_mode,
        ),
    )
    # Integrate each recorded action with full trajectory storage
    for a in actions:
        # Rebuild environment per action
        if pH_mode == "controlled":
            env = adapter.build_env_controlled_pH(a.get("pH_set", 0.0), a.get("stir", 0.0), a.get("temp", 37.0))
        elif pH_mode == "emergent":
            env = adapter.build_env_emergent_pH(a.get("stir", 0.0), a.get("temp", 37.0))
        else:  # switchable
            if int(a.get("pH_ctrl", 0)) == 1:
                env = adapter.build_env_controlled_pH(a.get("pH_set", 0.0), a.get("stir", 0.0), a.get("temp", 37.0))
            else:
                env = adapter.build_env_emergent_pH(a.get("stir", 0.0), a.get("temp", 37.0))
        pulse = adapter.create_pulse(q=float(a.get("q", 0.0)), v=float(a.get("v", 0.0)), env=env)
        adapter.integrate_pulse(pulse, store_states=True)

    r = adapter.reactor
    return {
        "time": getattr(r, "time_simul", None),
        "pH": getattr(r, "pH_simul", None),
        "met": getattr(r, "met_simul", None),  # shape (nmets, T)
        "subpop": getattr(r, "subpop_simul", None),  # shape (nsubpops, T)
        "species": getattr(r, "cellActive_dyn", None),  # dict name -> array
        "microbiome": r.microbiome,
        "metabolome": r.metabolome,
    }


def plot_performance(data: Dict[str, Any], outdir: Path, baseline: Optional[float] = None, title_suffix: str = "") -> None:
    _apply_pub_style()

    t = data["t"]
    target = data["target"]
    pH = data["pH"]
    reward = data["reward"]
    cum_reward = data["cum_reward"]
    target_kind = data.get("target_kind", "metabolite")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Target
    axes[0].plot(t, target, color="tab:orange", label=data.get("target_name", "target"))
    if baseline is not None:
        try:
            axes[0].axhline(float(baseline), color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline")
        except Exception:
            pass
    axes[0].set_ylabel("Target biomass" if str(target_kind).lower() == "biomass" else "Target [mM]")
    axes[0].set_title(f"Controller Performance{title_suffix}")
    axes[0].legend(loc="best")

    # pH
    axes[1].plot(t, pH, color="tab:blue", label="pH used")
    axes[1].set_ylabel("pH [-]")
    axes[1].legend(loc="best")

    # Rewards
    # Distribute last-step terminal spike across all steps for clearer visualization,
    # while preserving total episode reward. If no spike is detected, plot as-is.
    reward_vis = reward.copy()
    cum_vis = cum_reward.copy()
    if reward.size > 1:
        try:
            r_no_last = reward[:-1]
            med = float(np.median(np.abs(r_no_last)))
            std = float(np.std(r_no_last))
            thresh = max(10.0 * (med if med > 0 else 1e-6), 10.0 * (std if std > 0 else 1e-6))
            if abs(float(reward[-1])) > thresh:
                terminal_component = float(reward[-1])
                n = int(reward.size)
                share = terminal_component / float(n)
                reward_vis = reward.copy()
                reward_vis[-1] = 0.0
                reward_vis = reward_vis + share
                cum_vis = np.cumsum(reward_vis)
        except Exception:
            pass

    ma = _moving_average(reward_vis, max(1, len(reward_vis) // 100))
    axes[2].plot(t, reward_vis, color="tab:gray", alpha=0.4, label="Reward (step)")
    axes[2].plot(t, ma, color="tab:green", label="Reward (moving avg)")
    # Scale reward axis based on central percentiles to avoid flattening
    try:
        if reward_vis.size:
            lo = float(np.percentile(reward_vis, 1))
            hi = float(np.percentile(reward_vis, 99))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.05 * (hi - lo)
                axes[2].set_ylim(lo - pad, hi + pad)
    except Exception:
        pass
    ax2 = axes[2].twinx()
    ax2.plot(t, cum_vis, color="tab:red", label="Cumulative reward")
    axes[2].set_ylabel("Reward")
    ax2.set_ylabel("Cumulative")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="best")
    axes[-1].set_xlabel("Time [h]")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"performance.{ext}")
    plt.close(fig)


def plot_actions(data: Dict[str, Any], outdir: Path, title_suffix: str = "") -> None:
    _apply_pub_style()
    t = data["t"]
    actions = data["actions"]
    include_pH = bool(data.get("include_pH", True))

    # Build arrays
    keys = ["q", "v", "stir", "temp"]
    if include_pH:
        keys = ["q", "v", "pH_ctrl", "pH_set", "stir", "temp"]
    series = {k: np.array([float(a.get(k, np.nan)) for a in actions], dtype=np.float32) for k in keys}

    # rows: q, v, (optional pH_ctrl/pH_set), stir, temp
    nrows = 5 if include_pH else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
    # q
    axes[0].plot(t, series["q"], label="q [L/h]", color="tab:purple")
    axes[0].legend(loc="best"); axes[0].set_ylabel("q")
    # v
    axes[1].plot(t, series["v"], label="v [L]", color="tab:brown")
    axes[1].legend(loc="best"); axes[1].set_ylabel("v")
    row = 2
    if include_pH:
        ax = axes[row]
        ax2 = ax.twinx()
        ax.plot(t, series["pH_ctrl"], label="pH_ctrl", color="tab:gray")
        ax2.plot(t, series["pH_set"], label="pH_set", color="tab:blue")
        ax.set_ylabel("pH_ctrl"); ax2.set_ylabel("pH_set")
        lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        row += 1
    # stir
    axes[row].plot(t, series["stir"], label="stir", color="tab:olive")
    axes[row].legend(loc="best"); axes[row].set_ylabel("stir")
    row += 1
    # temp
    axes[row].plot(t, series["temp"], label="temp [C]", color="tab:red")
    axes[row].legend(loc="best"); axes[row].set_ylabel("temp")
    axes[-1].set_xlabel("Time [h]")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"actions.{ext}")
    plt.close(fig)


def plot_kinetics(
    kin: Dict[str, Any],
    outdir: Path,
    target_name: str,
    title_suffix: str = "",
    preferred_mets: Optional[List[str]] = None,
    top_mets: int = 4,
) -> None:
    _apply_pub_style()
    time = kin.get("time")
    met = kin.get("met")
    pH = kin.get("pH")
    subpop = kin.get("subpop")
    species = kin.get("species")  # dict name -> array
    metabolome = kin.get("metabolome")
    microbiome = kin.get("microbiome")

    if time is None:
        return
    t = np.asarray(time)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # (1) Metabolites: target + a few others
    ax = axes[0]
    try:
        names = list(getattr(metabolome, "metabolites", []))
        if isinstance(names, list) and names and isinstance(names[0], str):
            met_names = names
        else:
            # metabolome.metabolites might be a list of names; ModelFromJson sets .metabolites as list[str]
            met_names = list(metabolome.metabolites)
    except Exception:
        met_names = []
    plotted = set()
    # choose candidates
    candidates: List[str] = []
    if preferred_mets:
        candidates.extend([m for m in preferred_mets if m in met_names])
    if target_name in met_names and target_name not in candidates:
        candidates.insert(0, target_name)
    # auto-pick top others by max level
    if met is not None and met_names and top_mets > 0:
        try:
            arr = np.asarray(met)
            maxima = arr.max(axis=1)
            order = np.argsort(-maxima)
            for idx in order:
                name = met_names[int(idx)]
                if name not in candidates:
                    candidates.append(name)
                if len(candidates) >= max(top_mets, len(candidates)) and len(candidates) >= top_mets:
                    break
        except Exception:
            pass
    for name in candidates:
        try:
            i = met_names.index(name)
            ax.plot(t, met[i], label=name)
            plotted.add(name)
        except Exception:
            continue
    ax.set_title("Metabolites")
    ax.set_ylabel("[mM]")
    if plotted:
        ax.legend(loc="best")

    # (2) pH
    ax = axes[1]
    try:
        if pH is not None:
            ax.plot(t, pH, color="tab:blue", label="pH")
        ax.set_title("pH")
        ax.set_ylabel("pH [-]")
        ax.legend(loc="best")
    except Exception:
        ax.text(0.5, 0.5, "pH unavailable", transform=ax.transAxes, ha="center")

    # (3) Subpopulations (active only)
    ax = axes[2]
    try:
        subpop_idx = 0
        for bac_name, bac in microbiome.bacteria.items():
            for sp_name, sp in bac.subpopulations.items():
                if getattr(sp, "state", "active") != "active":
                    subpop_idx += 1
                    continue
                if subpop is not None and subpop_idx < subpop.shape[0]:
                    series = subpop[subpop_idx]
                    color = getattr(sp, "color", None)
                    ax.plot(t, series, label=f"{bac_name}_{sp_name}", color=color)
                subpop_idx += 1
        ax.set_title("Subpopulations (active)")
        ax.set_ylabel("Count / L")
        ax.legend(ncol=2, fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Subpops unavailable", transform=ax.transAxes, ha="center")

    # (4) Species (active only)
    ax = axes[3]
    try:
        if species:
            for species_name, series in species.items():
                bac = microbiome.bacteria.get(species_name)
                color = getattr(bac, "color", None) if bac is not None else None
                ax.plot(t, series, label=species_name, color=color, linewidth=2)
            ax.set_title("Species (active)")
            ax.set_ylabel("Count / L")
            ax.legend(ncol=2, fontsize=8)
        else:
            ax.text(0.5, 0.5, "Species dynamics unavailable", transform=ax.transAxes, ha="center")
    except Exception:
        ax.text(0.5, 0.5, "Species states unavailable", transform=ax.transAxes, ha="center")

    for ax in axes:
        ax.set_xlabel("Time [h]")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"kinetics.{ext}")
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Plot publication-quality figures for mg_rl_general controller runs")
    parser.add_argument("--model-json", type=str, default=str(base / "examples" / "bh_bt_ri_complete_model_export.json"))
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config for GeneralMicrobiomeEnv")
    parser.add_argument("--model-path", type=str, default=str(base / "models" / "mg_rl_general_sac.zip"))
    parser.add_argument("--algo", type=str, choices=["sac", "ppo", "tqc"], default="sac", help="Algorithm used to train the model")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device for the policy (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--norm-stats", type=str, default=None, help="Normalization stats JSON (auto-detect if omitted)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hours", type=float, default=250.0, help="Simulation duration in hours")
    parser.add_argument("--det", action="store_true", help="Use deterministic actions")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--baseline", type=float, default=None, help="Optional baseline target to overlay")
    parser.add_argument("--no-stats", action="store_true", help="Do not load saved normalization stats")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional title suffix for figures")
    parser.add_argument("--mets", type=str, default=None, help="Comma-separated metabolite names to plot (or 'all')")
    parser.add_argument("--top-mets", type=int, default=4, help="Number of top metabolites (by peak) to auto-include")
    parser.add_argument("--save-csv", action="store_true", help="Export step series and kinetics CSV files")
    parser.add_argument("--random-actions", action="store_true", help="Use random actions sampled from the environment action space instead of a trained model")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_json = str(args.model_json)
    if not args.random_actions:
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            sys.exit(1)
    if not Path(model_json).exists():
        print(f"Model JSON file not found: {model_json}")
        sys.exit(1)

    # Time-stamped output directory (avoid overwrites and enable comparisons)
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.out:
        base = Path(args.out)
        outdir = base.parent / f"{base.name}_{ts}"
    else:
        if args.random_actions:
            outdir = Path("backend/plots") / f"mg_rl_general_random_{ts}"
        else:
            outdir = Path("backend/plots") / f"mg_rl_general_{Path(model_path).stem}_{ts}"
    _ensure_outdir(outdir)

    # Load config and ensure horizon can cover requested hours
    cfg = _load_config(args.config)
    # Disable episode mix for a clean single-horizon rollout
    try:
        if cfg.episode.mix:
            cfg.episode.mix.update({"enabled": False})
    except Exception:
        pass
    # set horizon to at least the requested steps
    n_steps_needed = int(max(1, np.ceil(float(args.hours) / float(cfg.episode.dt_hours))))
    try:
        if cfg.episode.horizon < n_steps_needed:
            cfg.episode.horizon = int(n_steps_needed)
    except Exception:
        pass

    # Build environment and wrappers
    env_base = GeneralMicrobiomeEnv(model_json, cfg)
    env_wrapped = apply_observation_pipeline(env_base, cfg.observations.pipeline)

    # Load model (unless using random actions)
    model: Optional[Any]
    if args.random_actions:
        model = None
    else:
        if args.algo == "sac":
            model = SAC.load(str(model_path), device=str(args.device))
        elif args.algo == "ppo":
            model = PPO.load(str(model_path), device=str(args.device))
        elif args.algo == "tqc":
            if not _HAS_TQC:
                raise SystemExit("TQC model requested but sb3-contrib is not installed")
            model = TQC.load(str(model_path), device=str(args.device))  # type: ignore[operator]
        else:
            raise SystemExit(f"Unsupported algo: {args.algo}")

    # Load normalization stats if present (skip when random actions)
    if not args.random_actions:
        norm_path = Path(args.norm_stats) if args.norm_stats else None
        _try_load_norm_stats(env_wrapped, model_path, norm_path, enable=(not args.no_stats))

    # Run and collect
    data = _collect_run(model, env_wrapped, env_base, cfg, float(args.hours), bool(args.det), int(args.seed), random_actions=bool(args.random_actions))

    # Replay kinetics with stored trajectories
    kin = _replay_kinetics(model_json, cfg, data["actions"], env_base._pH_mode)

    # Titles
    if args.title_suffix:
        suffix = f" — {args.title_suffix}"
    else:
        if args.random_actions:
            suffix = f" — Random actions, T={int(args.hours)}h"
        else:
            suffix = f" — {Path(model_path).name}, T={int(args.hours)}h"

    # Plots
    plot_performance(data, outdir, baseline=args.baseline, title_suffix=suffix)
    plot_actions(data, outdir, title_suffix=suffix)
    # Determine preferred metabolite list
    preferred_mets: Optional[List[str]] = None
    top_mets = int(max(0, args.top_mets))
    if args.mets:
        s = args.mets.strip()
        if s.lower() == "all":
            # Plot all metabolites
            # Build names from model
            try:
                preferred_mets = list(env_base.adapter.metabolome.metabolites)
                top_mets = 0
            except Exception:
                preferred_mets = None
        else:
            preferred_mets = [m.strip() for m in s.split(",") if m.strip()]

    plot_kinetics(
        kin,
        outdir,
        target_name=data.get("target_name", "target"),
        title_suffix=suffix,
        preferred_mets=preferred_mets,
        top_mets=top_mets,
    )

    # Optional CSV exports
    if args.save_csv:
        # step-level series
        step_csv = outdir / "step_series.csv"
        with step_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            cols = ["t_hours","reward","cum_reward","pH_used","target","q","v","pH_ctrl","pH_set","stir","temp"]
            writer.writerow(cols)
            T = len(data["t"])
            actions = data["actions"]
            for i in range(T):
                a = actions[i] if i < len(actions) else {}
                writer.writerow([
                    float(data["t"][i]), float(data["reward"][i]), float(data["cum_reward"][i]), float(data["pH"][i]), float(data["target"][i]),
                    float(a.get("q", float("nan"))), float(a.get("v", float("nan"))), int(a.get("pH_ctrl", 0)), float(a.get("pH_set", float("nan"))),
                    float(a.get("stir", float("nan"))), float(a.get("temp", float("nan"))),
                ])

        # kinetics (time, pH, metabolites)
        if kin.get("time") is not None:
            # pH
            with (outdir / "kinetics_pH.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_hours","pH"])
                t = np.asarray(kin["time"]) ; ph = np.asarray(kin.get("pH")) if kin.get("pH") is not None else np.full_like(t, np.nan)
                for ti, pi in zip(t, ph):
                    w.writerow([float(ti), float(pi)])
            # metabolites
            met = kin.get("met")
            if met is not None and kin.get("metabolome") is not None:
                names = list(kin["metabolome"].metabolites)
                with (outdir / "kinetics_metabolites.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t_hours"] + names)
                    arr = np.asarray(met)  # (nmets, T)
                    t = np.asarray(kin["time"]).astype(float)
                    for j in range(arr.shape[1]):
                        row = [float(t[j])] + [float(arr[i, j]) for i in range(arr.shape[0])]
                        w.writerow(row)

    # Summary
    print("\nSummary:")
    print(f"  Steps: {len(data['t'])} (dt={data['dt']} h)")
    print(f"  Final {data.get('target_name','target')}: {data['target'][-1]:.4f}")
    print(f"  Final pH: {data['pH'][-1]:.3f}")
    print(f"  Total reward: {data['cum_reward'][-1]:.3f}")
    print(f"  Plots: {outdir}")


if __name__ == "__main__":
    main()








