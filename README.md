# Adaptive Diversity Regularization (ADR) for Cooperative MARL

This repository contains the reference implementation of **Adaptive Diversity
Regularization (ADR)**, a lightweight auxiliary action-diversity regularizer
that combats symmetric mode collapse in cooperative multi-agent reinforcement
learning without modifying the extrinsic reward.

ADR adds a single auxiliary loss term to the actor objective, with a
time-varying coefficient that is ramped up proactively at initialization,
attenuated through a soft ceiling tied to realized inter-agent diversity, and
withdrawn through either a reward-gated trigger or a time-gated truncation.
The three task-dependent knobs are derived in closed form from two observable
task statistics, removing the need for per-task grid search.

This code base supports running ADR on top of three actor-based MARL
algorithms (SEAC, MAA2C, MAPPO) on cooperative discrete-action benchmarks
(RWARE, LBF-coop, Overcooked-AI).

## Repository layout

```
.
├── train.py                  # main Sacred-based training entry point
├── run.py                    # multi-experiment launcher
├── a2c.py                    # per-agent A2C / SEAC base learner
├── ippo.py                   # independent PPO
├── maa2c.py                  # centralized-critic A2C (MAA2C)
├── mappo.py                  # centralized-critic clipped PPO (MAPPO)
├── pg_asnd.py                # ADR controller (eta(t) schedule, soft ceiling,
│                             #   reward-gated / time-gated shutoffs)
├── snd_monitor.py            # action-distribution diversity monitor
├── snd_rescale.py            # adaptive rescaling of diversity coefficient
├── snd_rescale_metrics.py    # diversity metrics for logging
├── state_diversity.py        # count-based state-visitation diversity (baseline)
├── state_div.py              # auxiliary state-diversity monitor
├── model.py                  # actor / critic networks
├── distributions.py          # categorical action heads
├── storage.py                # rollout storage
├── envs.py                   # vectorized environment construction
├── wrappers.py               # episode statistics, reward shaping, time limits
├── pettingzoo_wrapper.py     # PettingZoo bridge
├── evaluate.py               # post-training evaluation utility
├── utils.py                  # small utilities
├── configs/                  # Sacred config files for each task family
│   ├── rware*.yaml           # RWARE configs (tiny-2ag, small-4ag, etc.)
│   ├── foraging*.yaml        # LBF-coop configs (8x8-2p-{1,2}f, 12x12-2p-3f)
│   ├── overcooked*.yaml      # Overcooked-AI configs (forced_coordination,
│   │                           cramped_room, coordination_ring, ...)
│   └── pressureplate1.yaml   # PressurePlate (auxiliary)
├── overcooked_pkg/           # standalone gym-env adapter for Overcooked-AI
│   └── overcooked_seac/      # registers Overcooked-<layout>-v0 gym ids
├── requirements.txt          # pinned Python dependencies
└── constraints.txt           # pip constraint file
```

## Installation

The code targets Python 3.8. We recommend a fresh virtual environment.

```bash
python -m venv adr-env
source adr-env/bin/activate
pip install -U pip
pip install -r requirements.txt -c constraints.txt
```

The third-party MARL environments are installed transitively via the
requirements file:

* RWARE (`rware` / `robotic-warehouse`)
* LBF (`lbforaging`)
* PressurePlate (`pressureplate`, optional)

### Overcooked-AI setup

Overcooked-AI experiments rely on the
[`overcooked-ai`](https://github.com/HumanCompatibleAI/overcooked_ai) package
plus the lightweight gym-env adapter in `overcooked_pkg/`:

```bash
pip install overcooked-ai
pip install -e ./overcooked_pkg
```

Installing `overcooked_pkg` in editable mode registers a set of
`Overcooked-<layout>-v0` gym environment ids (e.g.
`Overcooked-forced_coordination-v0`,
`Overcooked-cramped_room-v0`,
`Overcooked-coordination_ring-v0`) that `train.py` consumes through the
same `envs.make_vec_envs` path as RWARE and LBF.

## Quick start

ADR is configured through Sacred. The same `train.py` entry point dispatches
to SEAC, MAA2C, MAPPO via `algorithm.name=...`.

```bash
# Baseline (no ADR), SEAC on RWARE small-4ag
python train.py with env_name=rware-small-4ag-v1 \
    algorithm.name=SEAC \
    algorithm.total_steps=40000000 \
    seed=42
```

```bash
# ADR-augmented run with the closed-form calibrated controller
python train.py with env_name=rware-small-4ag-v1 \
    algorithm.name=SEAC \
    algorithm.total_steps=40000000 \
    SND_DIVERSITY_COEF=1.0 \
    SND_PROACTIVE_RAMP_END=0.0375 \
    SND_PROACTIVE_SHUTOFF=1.00 \
    SND_PROACTIVE_REWARD_THRESH=1.0 \
    SND_PROACTIVE_MIN_PROGRESS=0.40 \
    SND_DIVERSITY_CEILING=0.50 \
    seed=42
```

```bash
# MAPPO on Overcooked-AI forced_coordination
python train.py with env_name=Overcooked-forced_coordination-v0 \
    algorithm.name=MAPPO \
    algorithm.total_steps=9000000 \
    SND_DIVERSITY_COEF=1.0 \
    SND_PROACTIVE_RAMP_END=0.05 \
    SND_PROACTIVE_TIME_SHUTOFF=0.95 \
    seed=42
```

The full list of ADR knobs and their per-experiment calibrated values is
given in Table 1 of the paper.

## Reproducing the paper

Each of the nine `<algorithm, environment>` experiments reported in the paper
corresponds to one Sacred configuration. The four sample-efficiency
experiments (SEAC and MAA2C on RWARE/LBF) use the closed-form calibration
described in Section 3.7 of the paper; the remaining configurations
(MAPPO on RWARE/LBF and the three Overcooked-AI experiments) use the open-loop
proactive boundary variant of the same controller with a single
`SND_PROACTIVE_TIME_SHUTOFF` value.

The calibrated knobs and base-learner hyperparameters listed in the paper are
sufficient to reproduce every run. Each run was launched with multiple
independent random seeds; we report the per-step mean across seeds with
mean ± SEM envelopes.

## Citing

A full BibTeX entry will be added once the paper is de-anonymized after
review. The current double-blind paper title is:

> *Adaptive Diversity Regularization for Cooperative Multi-Agent
> Reinforcement Learning.*

## License

This implementation is released for academic use. Third-party components
(SEAC, MAPPO, RWARE, LBF, Overcooked-AI, PettingZoo, stable-baselines3,
sacred) retain their respective original licenses.
