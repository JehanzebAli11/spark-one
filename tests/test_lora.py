"""LoRA trainer + HF backend against a tiny random Llama with a char-level tokenizer
(the real Gemma 4 path is exercised by scripts/smoke_gemma.py on real weights)."""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from spark_one.hf import HFBackend, encode_example, forward_example  # noqa: E402
from spark_one.prompt import question_text  # noqa: E402
from spark_one.schema import load_items  # noqa: E402
from spark_one.steps import item_steps  # noqa: E402
from spark_one.train import lora  # noqa: E402

TEMPLATE = ("{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{% if m['content'] is string %}{{ m['content'] }}"
            "{% else %}{% for c in m['content'] %}{% if c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}"
            "{% endif %}</s>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}")


def make_tok():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    chars = [chr(i) for i in range(32, 127)] + ["\n", ""] + list("āīūéè")
    vocab = {c: i for i, c in enumerate(["<pad>", "<s>", "</s>", "<unk>"] + chars)}
    t = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tok.add_tokens(["Yes", "No", "<system>", "<user>", "<assistant>"])
    tok.chat_template = TEMPLATE
    return tok


def make_loader():
    def loader(_):
        torch.manual_seed(0)
        tok = make_tok()
        cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
        return transformers.LlamaForCausalLM(cfg).float(), None, tok
    return loader


def test_hf_backend_prefix_cache_matches_full(small_bank):
    model, _, tok = make_loader()(None)
    be = HFBackend("tiny", model=model, processor=None, tok=tok)
    it = next(i for i in load_items(small_bank / "items.jsonl") if i.modality == "text" and i.primitive == "choice")
    ex = encode_example(it, small_bank, None, tok, be.device)
    with torch.no_grad():
        full = torch.log_softmax(forward_example(model, ex, be.device)[0, ex.step_label_ids[0]], -1)
    pre, _ = be.prefix(it.state, None)
    q, sp = question_text(it.primitive, it.instructions, it.criteria)
    cached = be.answer_steps(pre, q, sp, [s.labels for s in item_steps(it)])
    assert cached[0] == pytest.approx(full.tolist(), abs=1e-4)


def test_lora_train_and_resume(small_bank, tmp_path):
    def cfg(run):
        c = {"base_model": "tiny", "bank": str(small_bank), "run_dir": str(tmp_path / run), "steps": 4, "grad_accum": 2,
             "lr": 1e-3, "warmup": 1, "alpha": 0.5, "ckpt_every": 2, "log_every": 1, "modalities": ["text"],
             "lora": {"r": 4, "alpha": 8, "dropout": 0.0}}
        p = tmp_path / f"{run}.json"
        p.write_text(json.dumps(c))
        return str(p)

    lora.main(["--config", cfg("a")], model_loader=make_loader())
    assert (tmp_path / "a" / "final_adapter").exists()
    lora.main(["--config", cfg("b"), "--max_steps", "2"], model_loader=make_loader())
    lora.main(["--config", cfg("b"), "--resume_from", str(tmp_path / "b" / "checkpoint-2")], model_loader=make_loader())
    from safetensors.torch import load_file

    a = load_file(str(tmp_path / "a" / "final_adapter" / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "b" / "final_adapter" / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    logs = [json.loads(l) for l in (tmp_path / "a" / "train_log.jsonl").read_text().splitlines()]
    assert logs[-1]["step"] == 4
