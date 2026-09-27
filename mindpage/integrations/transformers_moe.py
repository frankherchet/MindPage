from __future__ import annotations

from typing import Any

from mindpage.tracing.models import ExpertChoice, TraceRun


class MissingMoEDependencies(RuntimeError):
    pass


class RouterLogitsUnavailable(RuntimeError):
    pass


def _require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingMoEDependencies(
            "MoE tracing requires the 'moe' extra: pip install -e '.[moe]'"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


class TransformersMoETracer:
    """Capture router choices exposed by Hugging Face causal MoE models."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> "TransformersMoETracer":
        torch, auto_model, auto_tokenizer = _require_dependencies()
        tokenizer = auto_tokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        model_dtype: Any = dtype
        if dtype != "auto":
            try:
                model_dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"unknown torch dtype: {dtype}") from exc
        model = auto_model.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return cls(model, tokenizer, model_id)

    def trace_prompt(
        self,
        prompt: str,
        *,
        task_group: str,
        max_length: int = 512,
        top_k: int | None = None,
    ) -> TraceRun:
        torch, _, _ = _require_dependencies()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(
                **encoded,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )

        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None:
            router_logits = getattr(outputs, "decoder_router_logits", None)
        if router_logits is None:
            raise RouterLogitsUnavailable(
                "model output did not expose router logits; choose a Transformers "
                "MoE architecture with output_router_logits support"
            )

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        token_text = tuple(self.tokenizer.convert_ids_to_tokens(input_ids))
        k = top_k or self._default_top_k()
        choices = self._extract_choices(router_logits, k)
        return TraceRun(
            model_id=self.model_id,
            task_group=task_group,
            prompt=prompt,
            token_ids=tuple(int(token_id) for token_id in input_ids),
            token_text=token_text,
            choices=choices,
        )

    def _default_top_k(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("num_experts_per_tok", "num_selected_experts", "top_k"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            value = getattr(text_config, "num_experts_per_tok", None)
            if isinstance(value, int) and value > 0:
                return value
        return 2

    @staticmethod
    def _extract_choices(router_logits: Any, top_k: int) -> tuple[ExpertChoice, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if hasattr(router_logits, "ndim"):
            layers = (router_logits,)
        else:
            layers = tuple(router_logits)

        choices: list[ExpertChoice] = []
        for layer_index, logits in enumerate(layers):
            tensor = logits.detach().float().cpu()
            if tensor.ndim == 3:
                tensor = tensor[0]
            if tensor.ndim != 2:
                raise RouterLogitsUnavailable(
                    f"unsupported router logit shape in layer {layer_index}: "
                    f"{tuple(tensor.shape)}"
                )
            if top_k > tensor.shape[-1]:
                raise ValueError(
                    f"top_k={top_k} exceeds {tensor.shape[-1]} experts in layer {layer_index}"
                )
            values, indices = tensor.topk(top_k, dim=-1)
            for token_index in range(tensor.shape[0]):
                for rank in range(top_k):
                    choices.append(
                        ExpertChoice(
                            layer=layer_index,
                            token_index=token_index,
                            expert_id=int(indices[token_index, rank].item()),
                            rank=rank,
                            score=float(values[token_index, rank].item()),
                        )
                    )
        return tuple(choices)
