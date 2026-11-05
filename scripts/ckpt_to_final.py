"""Turn a tiny-model checkpoint-N/ directory into a final.pt-style file for predict/serve."""
import json
import sys
from pathlib import Path

import torch

ck, out = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads((ck / "state.json").read_text())["config"]["model"]
from spark_one.tiny.model import TinyConfig  # noqa: E402

torch.save({"config": TinyConfig(**cfg).to_dict(), "state_dict": torch.load(ck / "model.pt")}, out)
print("wrote", out)
