from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Knobs:
    micro_batch: int
    grad_accum_steps: int


@dataclass
class ControllerState:
    best_knobs: Knobs | None = None
    best_throughput: float = 0.0
    in_trial: bool = False
    rl_algo: Any | None = None
    rl_action_space: Any | None = None
    rl_state_builder: Any | None = None
    rl_window: Any | None = None
    rl_safety: Any | None = None
    rl_knob_config: Any | None = None
    rl_modules: dict[str, Any] = field(default_factory=dict)
    last_action_name: str = ""
    last_safety_allowed: bool = True
    last_safety_reason: str = ""
