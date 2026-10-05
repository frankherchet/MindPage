from mindpage.tracing import ExpertChoice, TraceRun, jaccard, summarize_task_groups


def run(task_group: str, experts: list[int]) -> TraceRun:
    choices = tuple(
        ExpertChoice(layer=0, token_index=index, expert_id=expert, rank=0, score=1.0)
        for index, expert in enumerate(experts)
    )
    return TraceRun(
        model_id="fake",
        task_group=task_group,
        prompt="prompt",
        token_ids=tuple(range(len(experts))),
        token_text=tuple("x" for _ in experts),
        choices=choices,
    )


def test_jaccard() -> None:
    assert jaccard([(0, 1), (0, 2)], [(0, 2), (0, 3)]) == 1 / 3


def test_group_summary_reports_stability() -> None:
    summary = summarize_task_groups(
        [run("cpp", [1, 1, 2]), run("cpp", [1, 1, 3])], top_n=2
    )
    assert summary["cpp"]["runs"] == 2
    assert summary["cpp"]["tokens"] == 6
    assert summary["cpp"]["mean_pairwise_jaccard"] == 1 / 3
    assert summary["cpp"]["top_experts"][0]["expert_id"] == 1
