"""Hermetic fixtures: a tiny random Qwen2 model and a word-level tokenizer with a ChatML
template, so the full train -> decode path runs in CI without downloading weights."""

from __future__ import annotations

import pytest
import torch

CHAT = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

WORDS = (
    "card arrival lost or stolen exchange rate top up pending cash withdrawal my has not arrived yet "
    "i want to change pin the is where how do what classify bank support message's intent . : ? , "
    "system user assistant labelled messages similar new one candidate intents answer with name only "
    "you customer sent a digital chat into exactly allowed as written ; - message intent: tiny"
).split()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    special = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    vocab = {t: i for i, t in enumerate(special + sorted(set(WORDS)) + ["\n"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit()])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = CHAT
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        initializer_range=0.2,
        eos_token_id=vocab["<|im_end|>"],
        pad_token_id=vocab["<pad>"],
    )
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)
