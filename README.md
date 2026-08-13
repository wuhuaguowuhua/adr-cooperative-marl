# Adaptive Diversity Regularization (ADR) for Cooperative MARL

This repository contains the reference implementation of **Adaptive Diversity
Regularization (ADR)**, a lightweight auxiliary action-diversity regularizer
that combats symmetric mode collapse in cooperative multi-agent reinforcement
learning without modifying the extrinsic reward.

ADR adds a single auxiliary loss term to the actor objective, with a
time-varying coefficient that is ramped up proactively at initialization,
attenuated through a soft ceiling tied to realized inter-agent diversity, and
withdrawn through either a reward-gated trigger or a time-gated truncation.
Most closed-loop settings initialize three task-dependent knobs from two
statistics of a short baseline pilot. Documented guard and open-loop variants
cover boundary cases. This protocol reduces, rather than eliminates,
task-specific calibration.

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
├── snd_monitor.py            # ADR controller, diversity monitor, and schedules
├── pg_asnd.py                # legacy policy-gradient diversity controller
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
├── launch_p0_rware.sh        # four-arm SEAC + RWARE schedule comparison
├── launch_p0_lbf.sh          # four-arm MAA2C + LBF schedule comparison
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

Sacred named configs select the task. The `ALGO` environment variable selects
the learner: `a2c` (default, SEAC update), `maa2c`, `mappo`, or `ippo`.
ADR itself is configured through `SND_*` environment variables.

```bash
# Baseline (no ADR), SEAC on RWARE small-4ag
ALGO=a2c SND_ENABLE=0 python train.py with paper_rware_seac \
    algorithm.seed=42
```

```bash
# ADR on the same task
ALGO=a2c \
SND_ENABLE=1 SND_LOSS_MODE=rcdc SND_TRIGGER=feedback SND_METRIC=l2 \
SND_DIVERSITY_COEF=0.1 SND_ETA_MAX=0.10 \
SND_WARMUP_RATIO=0.0375 \
SND_PROACTIVE_RAMP_END=0.225 SND_PROACTIVE_RAMP_POWER=2.0 \
SND_PROACTIVE_TIME_SHUTOFF=1.0 SND_DIVERSITY_CEILING=1.0 \
SND_FEEDBACK_MIN_PEAK=1.0 SND_FEEDBACK_MIN_PROGRESS=0.40 \
python train.py with paper_rware_seac algorithm.seed=42
```

Set `WANDB_PROJECT`, `WANDB_RUN_GROUP`, and `WANDB_RUN_NAME` to organize
optional W&B logging. Authentication is handled by `wandb login`; no
credential is required by the launch scripts.

The full list of ADR knobs, including documented boundary variants, is given
in the calibration table of the paper.

## Controlled schedule ablation

The post-review comparison separates the ADR controller from the auxiliary
diversity loss using four matched arms:

| Arm | Setting |
| --- | --- |
| Baseline | `SND_ENABLE=0` |
| Static | `SND_TRIGGER=static`, `eta(t) = eta_max` |
| Linear | `SND_TRIGGER=linear`, `eta(t) = eta_max * (1 - progress)` |
| ADR | `SND_TRIGGER=feedback`, with the paper's calibrated controller |

The two reported matched comparisons can be reproduced directly:

```bash
# SEAC + rware-small-4ag-v1, 40M steps, seeds 42/123/456
WANDB_PROJECT=adr-p0-rware ./launch_p0_rware.sh

# MAA2C + Foraging-8x8-2p-1f-coop-v1, 40M steps, seeds 42/123/456
WANDB_PROJECT=adr-p0-lbf ./launch_p0_lbf.sh
```

The scripts run sequentially by default to avoid oversubscribing shared
machines. Runs can be restricted without editing the scripts:

```bash
# Smoke-test one arm and one seed
ARMS=static SEEDS=42 WANDB_DISABLED=true ./launch_p0_rware.sh

# Print commands without starting training
DRY_RUN=1 ./launch_p0_lbf.sh
```

Static and linear bypass all feedback, ceiling, ramp, and shutoff logic while
using the same peak coefficient as ADR. The named configs
`paper_rware_seac` and `paper_lbf_maa2c` fix the exact environments, horizon,
and training budget used in the controlled comparison.

## Reproducing the paper

The paper evaluates nine algorithm--task configurations across SEAC, MAA2C,
MAPPO, RWARE, LBF-coop, and Overcooked-AI. Most closed-loop rows initialize
the adaptive knobs from a baseline pilot; reward-guard and open-loop boundary
variants are listed explicitly in the paper's calibration table.

Use `ALGO` for the learner, a named config (or an explicit `env_name`
override) for the task, and the listed `SND_*` values for ADR. Baseline and
ADR comparisons use paired seed sets. Sample efficiency is normalized reward
area under the complete learning curve (N-AUC), and final reward is averaged
over the last two million environment steps. Paper curves use a centered
one-million-step moving average and mean ± SEM envelopes.

## Citing

A full BibTeX entry will be added once the paper is de-anonymized after
review. The current double-blind paper title is:

> *Adaptive Diversity Regularization for Cooperative Multi-Agent
> Reinforcement Learning.*

## License

This implementation is released for academic use. Third-party components
(SEAC, MAPPO, RWARE, LBF, Overcooked-AI, PettingZoo, stable-baselines3,
sacred) retain their respective original licenses.
