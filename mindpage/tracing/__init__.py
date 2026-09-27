"""Model-router tracing and analysis primitives."""

from .analysis import expert_counts, jaccard, summarize_task_groups, top_expert_keys
from .io import PromptRecord, read_prompts, read_traces, write_traces
from .models import ExpertChoice, TraceRun

__all__ = [
    "ExpertChoice",
    "PromptRecord",
    "TraceRun",
    "expert_counts",
    "jaccard",
    "read_prompts",
    "read_traces",
    "summarize_task_groups",
    "top_expert_keys",
    "write_traces",
]
