"""Prompt templates. Every method sees the same task description so that differences
come from the method (labels in context, retrieved examples, or learned weights)."""

from __future__ import annotations

from .data import Example

TASK = "You classify customer messages sent to a digital bank's support chat into exactly one intent."


# Short system prompt for fine-tuned models (an explicit one also stops Qwen's chat template
# from inserting its long default system message).
FT_SYSTEM = "Classify the bank support message's intent."


def zero_shot(text: str, labels: list[str]) -> list[dict]:
    options = "\n".join(f"- {lab}" for lab in labels)
    system = f"{TASK}\nAllowed intents:\n{options}\nAnswer with the intent name only, exactly as written."
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def retrieval_few_shot(text: str, shots: list[Example], candidates: list[str] | None = None) -> list[dict]:
    """In-context examples retrieved from the labelled training pool (the 'RAG' arm)."""
    demo = "\n\n".join(f"Message: {s.text}\nIntent: {s.label}" for s in shots)
    cand = ""
    if candidates:
        cand = "\n\nCandidate intents: " + "; ".join(candidates)
    system = f"{TASK}\nAnswer with the intent name only."
    user = f"Labelled messages similar to the new one:\n\n{demo}{cand}\n\nNew message: {text}\nIntent:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def finetune(text: str) -> list[dict]:
    """Short prompt for the fine-tuned model: the label set lives in the weights."""
    return [{"role": "system", "content": FT_SYSTEM}, {"role": "user", "content": text}]
