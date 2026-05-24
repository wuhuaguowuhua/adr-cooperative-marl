"""Overcooked-AI wrapped as gym envs for the SEAC training pipeline.

Registers cooperative Overcooked layouts as multi-agent gym envs returning
the (obs_tuple, reward_list, done_list, info) 4-tuple convention shared
with our RWARE / LBF / BoxPushing wrappers.

Layouts chosen for the RCDC narrative:
  - asymmetric_advantages: clear specialization advantage (one side has
    onion supply, other has plates), perfect specialization-style task
    for diversity-driven role splitting.
  - cramped_room: tight 4x5 grid, imitation-style task (both agents
    must do similar things), useful as a sanity baseline.
  - coordination_ring: medium difficulty, requires turn-taking.
"""

from gym.envs.registration import register

_LAYOUTS = {
    "asymmetric_advantages":   dict(horizon=400),
    "cramped_room":            dict(horizon=400),
    "coordination_ring":       dict(horizon=400),
    "forced_coordination":     dict(horizon=400),
    "counter_circuit_o_1order":dict(horizon=400),
}

for _name, _kwargs in _LAYOUTS.items():
    register(
        id=f"Overcooked-{_name}-v0",
        entry_point="overcooked_seac.environment:OvercookedSEAC",
        kwargs={"layout_name": _name, **_kwargs},
    )
