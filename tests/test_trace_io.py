import json

from mindpage.tracing import ExpertChoice, TraceRun, read_prompts, read_traces, write_traces


def test_prompt_and_trace_roundtrip(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"task_group": "cpp", "prompt": "hello"}) + "\n")
    prompts = read_prompts(prompt_path)
    assert prompts[0].task_group == "cpp"

    trace = TraceRun(
        model_id="fake",
        task_group="cpp",
        prompt="hello",
        token_ids=(1,),
        token_text=("hello",),
        choices=(ExpertChoice(0, 0, 2, 0, 0.5),),
    )
    trace_path = tmp_path / "traces.jsonl"
    write_traces(trace_path, [trace])
    assert read_traces(trace_path) == (trace,)
