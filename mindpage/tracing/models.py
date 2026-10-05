from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExpertChoice:
    """One routed expert choice for one token in one MoE layer."""

    layer: int
    token_index: int
    expert_id: int
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class TraceRun:
    """Router decisions captured for one prompt forward pass."""

    model_id: str
    task_group: str
    prompt: str
    token_ids: tuple[int, ...]
    token_text: tuple[str, ...]
    choices: tuple[ExpertChoice, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def layer_count(self) -> int:
        return 0 if not self.choices else 1 + max(choice.layer for choice in self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_group": self.task_group,
            "prompt": self.prompt,
            "token_ids": list(self.token_ids),
            "token_text": list(self.token_text),
            "choices": [asdict(choice) for choice in self.choices],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRun":
        return cls(
            model_id=str(data["model_id"]),
            task_group=str(data["task_group"]),
            prompt=str(data["prompt"]),
            token_ids=tuple(int(value) for value in data["token_ids"]),
            token_text=tuple(str(value) for value in data["token_text"]),
            choices=tuple(ExpertChoice(**choice) for choice in data["choices"]),
        )
